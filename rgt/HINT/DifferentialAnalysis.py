import time

from math import ceil, floor
from scipy.stats import zscore
from scipy.stats import norm
from argparse import SUPPRESS

from multiprocessing import Pool, cpu_count

# Internal
from rgt.Util import ErrorHandler, HmmData
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.HINT.Util import *

import matplotlib.pyplot as plt

"""
Perform differential footprints analysis based on the prediction of transcription factor binding sites.

Authors: Zhijian Li
"""

dic = {"A": 0, "C": 1, "G": 2, "T": 3}


def diff_analysis_args(parser):
    # Input Options
    parser.add_argument("--organism", type=str, metavar="STRING", default="hg19",
                        help="Organism considered on the analysis. Must have been setup in the RGTDATA folder. "
                             "Common choices are hg19, hg38. mm9, and mm10. DEFAULT: hg19")
    parser.add_argument("--mpbs-files", metavar='FILE1,FILE2...', type=str,
                        help='Predicted motif binding sites for each condition.'
                             'Files should be separated with comma.')
    parser.add_argument("--reads-files", metavar='FILE1,FILE2...', type=str,
                        help='Reads for each condition. Files should be separated with comma.')
    parser.add_argument("--conditions", metavar='STRING', type=str,
                        help='Name for each condition. DEFAULT: condition1,condition2, ...')
    parser.add_argument("--colors", metavar='STRING', type=str,
                        help='Set color in line plot. DEFAULT: None, ...')
    parser.add_argument("--window-size", type=int, metavar="INT", default=200,
                        help="The window size for differential analysis. DEFAULT: 200")

    parser.add_argument("--fdr", type=float, metavar="FLOAT", default=0.05,
                        help="The false discovery rate. DEFAULT: 0.05")
    parser.add_argument("--bc", action="store_true", default=False,
                        help="If set, all analysis will be based on bias corrected signal. DEFAULT: False")
    parser.add_argument("--nc", type=int, metavar="INT", default=1,
                        help="The number of cores. DEFAULT: 1")

    parser.add_argument("--forward-shift", type=int, metavar="INT", default=5, help=SUPPRESS)
    parser.add_argument("--reverse-shift", type=int, metavar="INT", default=-4, help=SUPPRESS)

    # Output Options
    parser.add_argument("--output-location", type=str, metavar="PATH", default=os.getcwd(),
                        help="Path where the output bias table files will be written. DEFAULT: current directory")
    parser.add_argument("--output-prefix", type=str, metavar="STRING", default="differential",
                        help="The prefix for results files. DEFAULT: differential")
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="If set, the signal will be rescaled to (0, 1) for plotting.")
    parser.add_argument("--output-profiles", default=False, action='store_true',
                        help="If set, the footprint profiles will be writen into a text, in which each row is a "
                             "specific instance of the given motif. DEFAULT: False")


def diff_analysis_run(args):
    # Initializing Error Handler
    err = ErrorHandler()

    output_location = os.path.join(args.output_location, "Lineplots")
    if not os.path.isdir(output_location):
        os.makedirs(output_location)

    print("{} cpus are detected and {} of them will be used...\n".format(cpu_count(), args.nc))

    # check if they have same length
    mpbs_files = args.mpbs_files.strip().split(",")
    reads_files = args.reads_files.strip().split(",")
    conditions = args.conditions.strip().split(",")

    if args.colors is not None:
        colors = args.colors.strip().split(",")
    else:
        colors = ["#000000", "#000099", "#006600", "#990000", "#660099", "#CC00CC", "#222222", "#CC9900",
                  "#FF6600", "#0000CC", "#336633", "#CC0000", "#6600CC", "#FF00FF", "#555555", "#CCCC00",
                  "#FF9900", "#0000FF", "#33CC33", "#FF0000", "#663399", "#FF33FF", "#888888", "#FFCC00",
                  "#663300", "#009999", "#66CC66", "#FF3333", "#9933FF", "#FF66FF", "#AAAAAA", "#FFCC33",
                  "#993300", "#00FFFF", "#99FF33", "#FF6666", "#CC99FF", "#FF99FF", "#CCCCCC", "#FFFF00"]

    assert len(mpbs_files) == len(reads_files) == len(conditions), \
        "Number of motif matching files, bam files and condition names are not same: {}, {}, {}".format(
            len(mpbs_files), len(reads_files), len(conditions))

    # Check if the index file exists
    for reads_file in reads_files:
        base_name = "{}.bai".format(reads_file)
        if not os.path.exists(base_name):
            pysam.index(reads_file)

    print("{}: loading motif matching files...\n".format(time.strftime("%D-%H:%M:%S")))
    mpbs = GenomicRegionSet("Motif Predicted Binding Sites of All Conditions")
    for i, mpbs_file in enumerate(mpbs_files):
        mpbs.read(mpbs_file)

    mpbs.sort()
    mpbs.remove_duplicates()
    mpbs_name_list = list(set(mpbs.get_names()))

    motif_len = list()
    motif_num = list()
    motif_pwm = list()

    print("{}: preparing data for differential footprinting analysis...\n".format(time.strftime("%D-%H:%M:%S")))
    mpbs_regions_by_name = dict()
    for region in mpbs:
        if region.name not in mpbs_regions_by_name:
            mpbs_regions_by_name[region.name] = GenomicRegionSet(region.name)
        mpbs_regions_by_name[region.name].add(region)

    del mpbs

    for mpbs_name in mpbs_name_list:
        motif_len.append(mpbs_regions_by_name[mpbs_name][0].final - mpbs_regions_by_name[mpbs_name][0].initial)
        motif_num.append(len(mpbs_regions_by_name[mpbs_name]))

    print("{}: generating pwm for each factor...\n".format(time.strftime("%D-%H:%M:%S")))
    if args.nc == 1:
        for mpbs_name in mpbs_name_list:
            pwm = get_pwm([args.organism, mpbs_regions_by_name[mpbs_name], args.window_size])
            motif_pwm.append(pwm)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for mpbs_name in mpbs_name_list:
                arguments_list.append([args.organism, mpbs_regions_by_name[mpbs_name], args.window_size])

            pwm_list = pool.map(get_pwm, arguments_list)
            for i, mpbs_name in enumerate(mpbs_name_list):
                motif_pwm.append(pwm_list[i])

    signals = np.zeros(shape=(len(conditions), len(mpbs_name_list), args.window_size), dtype=np.float32)
    # differential analysis using bias corrected signal
    if args.bc:
        print("{}: loading bias table for bias correction...\n".format(time.strftime("%D-%H:%M:%S")))
        hmm_data = HmmData()
        table_forward = hmm_data.get_default_bias_table_F_ATAC()
        table_reverse = hmm_data.get_default_bias_table_R_ATAC()
        bias_table = load_bias_table(table_file_name_f=table_forward, table_file_name_r=table_reverse)

        # do not use multi-processing
        if args.nc == 1:
            for i, condition in enumerate(conditions):
                print("{}: generating signal for condition {} ...\n".format(time.strftime("%D-%H:%M:%S"),
                                                                            condition))
                for j, mpbs_name in enumerate(mpbs_name_list):
                    mpbs_regions = mpbs_regions_by_name[mpbs_name]
                    arguments = (mpbs_regions, reads_files[i], args.organism, args.window_size, args.forward_shift,
                                 args.reverse_shift, bias_table)

                    signal = get_bc_signal(arguments)
                    if not np.isnan(signal).any():
                        signals[i, j, :] = get_bc_signal(arguments)

        # use multi-processing
        else:
            for i, condition in enumerate(conditions):
                print("{}: generating signal for condition {} ...\n".format(time.strftime("%D-%H:%M:%S"),
                                                                            condition))
                with Pool(processes=args.nc) as pool:
                    arguments_list = list()
                    for mpbs_name in mpbs_name_list:
                        mpbs_regions = mpbs_regions_by_name[mpbs_name]
                        arguments = (mpbs_regions, reads_files[i], args.organism, args.window_size, args.forward_shift,
                                     args.reverse_shift, bias_table)
                        arguments_list.append(arguments)

                    res = pool.map(get_bc_signal, arguments_list)
                    signals[i] = np.array(res)

    # differential analysis using raw signal
    else:
        # do not use multi-processing
        if args.nc == 1:
            for i, condition in enumerate(conditions):
                print("{}: generating signal for condition {} ...\n".format(time.strftime("%D-%H:%M:%S"),
                                                                            condition))
                for j, mpbs_name in enumerate(mpbs_name_list):
                    mpbs_regions = mpbs_regions_by_name[mpbs_name]
                    arguments = (mpbs_regions, reads_files[i], args.organism, args.window_size, args.forward_shift,
                                 args.reverse_shift)

                    signals[i, j, :] = get_raw_signal(arguments)

        # use multi-processing
        else:
            for i, condition in enumerate(conditions):
                print("{}: generating signal for condition {} ...\n".format(time.strftime("%D-%H:%M:%S"),
                                                                            condition))
                with Pool(processes=args.nc) as pool:
                    arguments_list = list()
                    for mpbs_name in mpbs_name_list:
                        mpbs_regions = mpbs_regions_by_name[mpbs_name]
                        arguments = (mpbs_regions, reads_files[i], args.organism, args.window_size, args.forward_shift,
                                     args.reverse_shift)
                        arguments_list.append(arguments)

                    res = pool.map(get_raw_signal, arguments_list)
                    signals[i] = np.array(res)

    print("{}: signal generation is done!".format(time.strftime("%D-%H:%M:%S")))

    # compute normalization facotr for each condition
    factors = compute_factors(signals)
    output_factor(args, factors, conditions)

    # normalize signals by factor and number of motifs
    for i in range(len(conditions)):
        for j in range(len(mpbs_name_list)):
            signals[i, j, :] = signals[i, j, :] / (factors[i] * motif_num[j])

    if args.output_profiles:
        output_profiles(mpbs_name_list, signals, conditions, args.output_location)

    print("{}: generating line plot for each motif...\n".format(time.strftime("%D-%H:%M:%S")))
    if args.nc == 1:
        for i, mpbs_name in enumerate(mpbs_name_list):
            output_line_plot_multi_conditions(
                (mpbs_name, motif_num[i], signals[:, i, :], conditions, motif_pwm[i], output_location,
                 args.window_size, colors))
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for i, mpbs_name in enumerate(mpbs_name_list):
                arguments_list.append(
                    (mpbs_name, motif_num[i], signals[:, i, :], conditions, motif_pwm[i], output_location,
                     args.window_size, colors))
            pool.map(output_line_plot_multi_conditions, arguments_list)

    ps_tc_results = list()
    for i, mpbs_name in enumerate(mpbs_name_list):
        ps_tc_results.append(get_ps_tc_results(signals[:, i, :], motif_len[i], args.window_size))

    # find the significant motifs and generate a scatter plot if two conditions are given
    if len(conditions) == 2:
        ps_tc_results = scatter_plot(args, ps_tc_results, mpbs_name_list, conditions)

    output_stat_results(ps_tc_results, conditions, mpbs_name_list, motif_num, args)


def get_raw_signal(arguments):
    (mpbs_region, reads_file, organism, window_size, forward_shift, reverse_shift) = arguments

    bam = pysam.Samfile(reads_file, "rb")
    signal = np.zeros(window_size)

    for region in mpbs_region:
        mid = (region.final + region.initial) // 2
        p1 = mid - window_size // 2
        p2 = mid + window_size // 2

        if p1 <= 0:
            continue
        # Fetch raw signal
        for read in bam.fetch(region.chrom, p1, p2):
            # check if the read is unmapped, according to issue #112
            if read.is_unmapped:
                continue

            if not read.is_reverse:
                cut_site = read.pos + forward_shift
                if p1 <= cut_site < p2:
                    signal[cut_site - p1] += 1.0
            else:
                cut_site = read.aend + reverse_shift - 1
                if p1 <= cut_site < p2:
                    signal[cut_site - p1] += 1.0

    return signal


def get_bc_signal(arguments):
    (mpbs_region, reads_file, organism, window_size, forward_shift, reverse_shift, bias_table) = arguments

    bam = pysam.Samfile(reads_file, "rb")
    genome_data = GenomeData(organism)
    signal = np.zeros(window_size)
    # Fetch bias corrected signal
    for region in mpbs_region:
        mid = int((region.final + region.initial) / 2)
        p1 = int(mid - window_size / 2)
        p2 = int(mid + window_size / 2)

        if p1 <= 0:
            continue
        # Fetch raw signal
        _signal = bias_correction(chrom=region.chrom, start=p1, end=p2, bam=bam,
                                  bias_table=bias_table, genome_file_name=genome_data.get_genome(),
                                  forward_shift=forward_shift, reverse_shift=reverse_shift)
        if len(_signal) != window_size:
            continue

        # smooth the signal
        signal = np.add(signal, np.array(_signal))

    return signal


def bias_correction(chrom, start, end, bam, bias_table, genome_file_name, forward_shift, reverse_shift):
    # Parameters
    window = 50
    default_kmer_value = 1.0

    # Initialization
    fastaFile = pysam.Fastafile(genome_file_name)
    fBiasDict = bias_table[0]
    rBiasDict = bias_table[1]
    k_nb = len(list(fBiasDict.keys())[0])
    p1 = start
    p2 = end
    p1_w = int(p1 - (window / 2))
    p2_w = int(p2 + (window / 2))
    p1_wk = p1_w - int(floor(k_nb / 2.))
    p2_wk = p2_w + int(ceil(k_nb / 2.))
    if p1 <= 0 or p1_w <= 0 or p1_wk <= 0 or p2_wk <= 0:
        # Return raw counts
        bc_signal = np.zeros(p2 - p1)
        for read in bam.fetch(chrom, p1, p2):
            # check if the read is unmapped, according to issue #112
            if read.is_unmapped:
                continue

            if not read.is_reverse:
                cut_site = read.pos + forward_shift
                if p1 <= cut_site < p2:
                    bc_signal[cut_site - p1] += 1.0
            else:
                cut_site = read.aend + reverse_shift - 1
                if p1 <= cut_site < p2:
                    bc_signal[cut_site - p1] += 1.0

        return bc_signal

    # Raw counts
    nf = np.zeros(p2_w - p1_w)
    nr = np.zeros(p2_w - p1_w)
    for read in bam.fetch(chrom, p1_w, p2_w):
        # check if the read is unmapped, according to issue #112
        if read.is_unmapped:
            continue

        if not read.is_reverse:
            cut_site = read.pos + forward_shift
            if p1_w <= cut_site < p2_w:
                nf[cut_site - p1_w] += 1.0
        else:
            cut_site = read.aend + reverse_shift - 1
            if p1_w <= cut_site < p2_w:
                nr[cut_site - p1_w] += 1.0

    # Smoothed counts
    Nf = []
    Nr = []
    f_sum = sum(nf[:window])
    r_sum = sum(nr[:window])
    f_last = nf[0]
    r_last = nr[0]
    for i in range(int((window / 2)), len(nf) - int((window / 2))):
        Nf.append(f_sum)
        Nr.append(r_sum)
        f_sum -= f_last
        f_sum += nf[i + int((window / 2))]
        f_last = nf[i - int((window / 2)) + 1]
        r_sum -= r_last
        r_sum += nr[i + int((window / 2))]
        r_last = nr[i - int((window / 2)) + 1]

    # Fetching sequence
    currStr = str(fastaFile.fetch(chrom, p1_wk, p2_wk - 1)).upper()
    currRevComp = AuxiliaryFunctions.revcomp(str(fastaFile.fetch(chrom, p1_wk + 1, p2_wk)).upper())

    # Iterating on sequence to create signal
    af = []
    ar = []
    for i in range(int(ceil(k_nb / 2.)), len(currStr) - int(floor(k_nb / 2)) + 1):
        fseq = currStr[i - int(floor(k_nb / 2.)):i + int(ceil(k_nb / 2.))]
        rseq = currRevComp[len(currStr) - int(ceil(k_nb / 2.)) - i:len(currStr) + int(floor(k_nb / 2.)) - i]
        try:
            af.append(fBiasDict[fseq])
        except Exception:
            af.append(default_kmer_value)
        try:
            ar.append(rBiasDict[rseq])
        except Exception:
            ar.append(default_kmer_value)

    # Calculating bias and writing to wig file
    f_sum = sum(af[:window])
    r_sum = sum(ar[:window])
    f_last = af[0]
    r_last = ar[0]
    bc_signal = []
    for i in range(int((window / 2)), len(af) - int((window / 2))):
        nhatf = Nf[i - int((window / 2))] * (af[i] / f_sum)
        nhatr = Nr[i - int((window / 2))] * (ar[i] / r_sum)
        bc_signal.append(nhatf + nhatr)
        f_sum -= f_last
        f_sum += af[i + int((window / 2))]
        f_last = af[i - int((window / 2)) + 1]
        r_sum -= r_last
        r_sum += ar[i + int((window / 2))]
        r_last = ar[i - int((window / 2)) + 1]

    # Termination
    fastaFile.close()
    return bc_signal


def get_ps_tc_results(signals, motif_len, window_size):
    signal_half_len = window_size // 2
    nc = np.sum(signals[:, int(signal_half_len - motif_len / 2):int(signal_half_len + motif_len / 2)], axis=1)
    nr = np.sum(signals[:, int(signal_half_len + motif_len / 2):int(signal_half_len + motif_len / 2 + motif_len)],
                axis=1)
    nl = np.sum(signals[:, int(signal_half_len - motif_len / 2 - motif_len):int(signal_half_len - motif_len / 2)],
                axis=1)

    protect_scores = (nr - nc) / motif_len + (nl - nc) / motif_len

    tcs = (np.sum(signals, axis=1) - nc) / (window_size - motif_len)

    return [protect_scores, tcs]


def compute_factors(signals):
    signals = np.sum(signals, axis=2)
    signals_log = np.log(signals)
    signals_log = signals_log[:, ~np.isnan(signals_log).any(axis=0)]
    signals_log = signals_log - np.mean(signals_log, axis=0, keepdims=True)
    factors = np.exp(np.median(signals_log, axis=1))

    return factors


def scatter_plot(args, ps_tc_results, mpbs_name_list, conditions):
    tf_activity_score1 = np.zeros(len(mpbs_name_list))
    tf_activity_score2 = np.zeros(len(mpbs_name_list))

    for i, mpbs_name in enumerate(mpbs_name_list):
        tf_activity_score1[i] = float(ps_tc_results[i][0][0]) + float(ps_tc_results[i][1][0])
        tf_activity_score2[i] = float(ps_tc_results[i][0][1]) + float(ps_tc_results[i][1][1])

    tf_activity_score = np.subtract(tf_activity_score2, tf_activity_score1)
    z_score = zscore(tf_activity_score)
    p_values = norm.sf(abs(z_score)) * 2

    # add TF activity score, z score and p values to the result dictionary
    for i, mpbs_name in enumerate(mpbs_name_list):
        ps_tc_results[i].append([tf_activity_score[i], z_score[i], p_values[i]])

    # plot TF activity score
    x_axis = np.random.uniform(low=-0.1, high=0.1, size=len(p_values))

    fig, ax = plt.subplots(figsize=(10, 12))
    for i, mpbs_name in enumerate(mpbs_name_list):
        if p_values[i] < args.fdr:
            ax.scatter(x_axis[i], tf_activity_score[i], c="red")
            ax.annotate(mpbs_name, (x_axis[i], tf_activity_score[i]), alpha=0.6)
        else:
            ax.scatter(x_axis[i], tf_activity_score[i], c="black", alpha=0.6)
    ax.margins(0.05)
    ax.set_xticks([])

    ax.set_ylabel("Activity Score \n {} $\longleftrightarrow$ {}".format(conditions[0], conditions[1]),
                  rotation=90, fontsize=20)

    figure_name = os.path.join(args.output_location, "{}_statistics.pdf".format(args.output_prefix))
    fig.savefig(figure_name, format="pdf", dpi=300)

    return ps_tc_results


def output_stat_results(ps_tc_results, conditions, mpbs_name_list, motif_num, args):
    output_filename = os.path.join(args.output_location, "{}_statistics.txt".format(args.output_prefix))

    if len(conditions) == 2:
        header = ["Motif", "Num",
                  "Protection_Score_{}".format(conditions[0]), "Protection_Score_{}".format(conditions[1]),
                  "TC_{}".format(conditions[0]), "TC_{}".format(conditions[1]), "TF_Activity", "Z_score", "P_values"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i, mpbs_name in enumerate(mpbs_name_list):
                f.write(mpbs_name + "\t" + str(motif_num[i]) + "\t" +
                        "\t".join(map(str, ps_tc_results[i][0])) + "\t" +
                        "\t".join(map(str, ps_tc_results[i][1])) + "\t" +
                        "\t".join(map(str, ps_tc_results[i][2])) + "\n")

    else:
        header = ["Motif", "Num"]
        for condition in conditions:
            header.append("Protection_Score_{}".format(condition))
        for condition in conditions:
            header.append("TC_{}".format(condition))

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i, mpbs_name in enumerate(mpbs_name_list):
                f.write(mpbs_name + "\t" + str(motif_num[i]) + "\t" +
                        "\t".join(map(str, ps_tc_results[i][0])) + "\t" +
                        "\t".join(map(str, ps_tc_results[i][1])) + "\n")


def output_factor(args, factors, conditions):
    output_file = os.path.join(args.output_location, "{}_factor.txt".format(args.output_prefix))
    with open(output_file, "w") as f:
        f.write("\t".join(conditions) + "\n")
        f.write("\t".join(map(str, factors)) + "\n")


def standard(vector1, vector2):
    max_ = max(max(vector1), max(vector2))
    min_ = min(min(vector1), min(vector2))
    if max_ > min_:
        return [(e - min_) / (max_ - min_) for e in vector1], [(e - min_) / (max_ - min_) for e in vector2]
    else:
        return vector1, vector2


def adjust_p_values(p_values):
    p = np.asfarray(p_values)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def output_profiles(mpbs_name_list, signals, conditions, output_location):
    for i, condition in enumerate(conditions):
        for j, mpbs_name in enumerate(mpbs_name_list):
            output_filename = os.path.join(output_location, "{}_{}.txt".format(condition, mpbs_name))
            with open(output_filename, "w") as f:
                f.write("\t".join(map(str, signals[i][j])) + "\n")
