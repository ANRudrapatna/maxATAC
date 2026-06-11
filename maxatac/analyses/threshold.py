import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, Manager
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank
from maxatac.utilities.plot import plot_threshold_calibration_stats
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import pybedtools
import logging

# Extract the bigWig values for both the prediction and the gold standard for the specified chromosome. 
# The bigWig signal is aggregated per bin.
def extract_pred_gs_bw(bigwig_file, training_data_dict, chrom_name, chrom_length, bin_count):
    start = time.time()
    bw_name = os.path.basename(bigwig_file)
    chrom_vals = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)
    print(training_data_dict[bigwig_file])
    
    predictions = np.empty(1, dtype = np.float64)
    gold_standard = np.empty(1, dtype = np.float64)
    goldstandard_array = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)
    
    tot_gs_bins = len(np.argwhere(goldstandard_array == True))

    predictions = np.concatenate([predictions, chrom_vals])
    gold_standard = np.concatenate([gold_standard, goldstandard_array])
    
    end = time.time()
    print('total time (s)= ' + str(end-start), "____________", bw_name)
    
    return predictions, gold_standard, tot_gs_bins

# Run thresholding.
def run_thresholding(args):
    """
    Generate a threshold file for the trained maxATAC model.
    :param args: output_dir, chromosomes, chrom_sizes, meta_file, prefix, bin_size, blacklist_bw, blacklist_bed
    :return: a CSV file containing 1) threshold values, 2) precision values, 3) recall values, and 4) F1 scores.
    """
    # Make the output directory and chromosome sizes dictionary.
    output_dir = get_dir(args.output_dir)
    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    # Read the input threshold meta file.
    meta_DF = pd.read_table(args.meta_file)
    training_data_dict = pd.Series(meta_DF["Binding_File"].values, index = meta_DF["Prediction"]).to_dict()
    results_filename = os.path.join(output_dir, args.prefix + ".tsv")

    # Loop through the chromosomes and average the values across files.
    OUT = []
    for chrom_name, chrom_length in chromosome_sizes_dictionary.items():
        bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number
        blacklist_mask = import_blacklist_mask(args.blacklist_bw, chrom_name, chrom_length, bin_count)
        blacklist = blacklist_mask
        
        lst_of_bws = list(training_data_dict.keys())   
        pool = Pool(int(multiprocessing.cpu_count())) 
        
        # Run the function defined above to extract the bigWig values for both the prediction and the gold standard.
        output = pool.starmap(extract_pred_gs_bw, [(bigwig, training_data_dict, chrom_name, chrom_length, bin_count) for bigwig in lst_of_bws])
        OUT.append(output)

    DF = pd.DataFrame([])
    total_gs_bins = []
    for i in range(len(OUT[0])):
        # Get the prediction and gold standard signal values, then concatenate them to a larger data frame.
        df = pd.DataFrame([])
        df['Prediction'] = OUT[0][i][0][:bin_count].tolist()
        df['GoldStandard'] = OUT[0][i][1][:bin_count].tolist()
        gs_bins = OUT[0][i][2]
        
        DF = pd.concat([DF, df], axis = 1, ignore_index = True)
        total_gs_bins.append(gs_bins)
    
    # Take the median of the prediction and gold standard signals across all cell types.
    DF_median = pd.DataFrame([])
    DF_median['Prediction'] = np.nanmedian(DF[range(0, np.shape(DF)[1], 2)], axis = 1)
    DF_median['GoldStandard'] = np.nanmedian(DF[range(1, np.shape(DF)[1], 2)], axis = 1)
    
    # Remove regions where there is no signal from the gold standard, then calculate precision-recall statistics.
    DF_median.loc[DF_median['GoldStandard'] != 1, 'GoldStandard'] = 0
    precision, recall, thresholds = precision_recall_curve(DF_median['GoldStandard'][blacklist], DF_median['Prediction'][blacklist])
    
    # Create a dataframe from the results. Extend the last row of entries to a threshold of 1.
    PR_CURVE_DF = pd.DataFrame({'Precision': precision, 'Recall': recall, "Threshold": np.insert(thresholds, 0, 0)})
    new_row = PR_CURVE_DF.tail(n = 1)
    new_row.Threshold = 1
    PR_CURVE_DF = pd.concat([PR_CURVE_DF, new_row], ignore_index = True)
    
    # Define a new set of 102 evenly-spaced threshold values using the maximum median prediction value across all cell types.
    # Then identify all rows in scikit-learn's precision_recall_curve output (line 87) with threshold values 
    # greater than each "threshold bin". Keep the row which maximizes recall for every threshold bin.
    threshold_values = np.arange(0, max(DF_median['Prediction'].unique()), max(DF_median['Prediction'].unique())/102)
    PR_CURVE_DF_THRESHOLD = pd.DataFrame([])
    for threshold in threshold_values:
        df_tmp = PR_CURVE_DF[PR_CURVE_DF['Threshold'] >= threshold]
        row = df_tmp[df_tmp['Recall'] == max(df_tmp['Recall'])].reset_index().head(1)
        PR_CURVE_DF_THRESHOLD = pd.concat([PR_CURVE_DF_THRESHOLD, row], ignore_index = True)
        
    # Take the cumulative sum along the precision and recall entries.    
    PR_CURVE_DF_THRESHOLD['Precision'] = np.maximum.accumulate((np.array(PR_CURVE_DF_THRESHOLD['Precision'])))
    PR_CURVE_DF_THRESHOLD['Recall'] = np.minimum.accumulate((np.array(PR_CURVE_DF_THRESHOLD['Recall'])))
    
    # Delete unneeded columns.
    del(PR_CURVE_DF_THRESHOLD['index'])
    del(PR_CURVE_DF_THRESHOLD['Threshold'])
    
    # total_median_gs_bins: Median GS bins across all cell tyoes.
    PR_CURVE_DF_THRESHOLD["Total_Median_GoldStandard_Bins"] = int(np.median(total_gs_bins))
    
    # Create a BEDtools object that is a windowed genome. Then create a blacklist object from the blacklist BED file.
    # Next, remove the blacklisted regions from the windowed genome object and create a dataframe from the BEDtools object.
    BED_df_bedtool = pybedtools.BedTool().window_maker(g = args.chrom_sizes, w = args.bin_size)
    blacklist_bedtool = pybedtools.BedTool(args.blacklist_bed)
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v = True)
    df = blacklisted_df.to_dataframe()
    
    # Rename the columns.
    df.columns = ["chr", "start", "stop"]
    
    # Find the number of non-blacklisted bins in the chr of interest.
    rand_bins = df.query('chr == @args.chromosomes').shape[0]
    
    # Calculate the following metrics: random precision, log2FC of (precision/random precision) -- a pseudocount of 0.1 has been added, F1-score    
    logging.info("Calculate log2FC for each threshold.")   
    PR_CURVE_DF_THRESHOLD['Random_Precision'] = PR_CURVE_DF_THRESHOLD['Total_Median_GoldStandard_Bins']/rand_bins
    PR_CURVE_DF_THRESHOLD['log2FC_Precision_Random_Precision'] = np.log2((PR_CURVE_DF_THRESHOLD["Precision"] + 0.1) /(PR_CURVE_DF_THRESHOLD["Random_Precision"] + 0.1))
    
    logging.info("Calculate F1 Score for each threshold")
    PR_CURVE_DF_THRESHOLD['F1_Score'] = (2 * (PR_CURVE_DF_THRESHOLD["Precision"] * PR_CURVE_DF_THRESHOLD["Recall"])) / (PR_CURVE_DF_THRESHOLD["Precision"] + PR_CURVE_DF_THRESHOLD["Recall"])
    PR_CURVE_DF_THRESHOLD['Max_F1_Score'] = max(PR_CURVE_DF_THRESHOLD['F1_Score'])
    
    # Delete unneeded columns.
    del PR_CURVE_DF_THRESHOLD['Total_Median_GoldStandard_Bins']
    del PR_CURVE_DF_THRESHOLD['Random_Precision']
    
    # Rename and reorder the columns of the data frame. Add a column for the standard threshold.
    PR_CURVE_DF_THRESHOLD.columns = ['Monotonic_Median_Precision', 'Monotonic_Median_Recall', 'Monotonic_Median_log2FC', 'F1_Score', 'Max_F1_Score']
    PR_CURVE_DF_THRESHOLD = PR_CURVE_DF_THRESHOLD.drop(PR_CURVE_DF_THRESHOLD.index[-1])
    PR_CURVE_DF_THRESHOLD['Standard_Threshold'] = np.round(np.arange(0, 1.01, 0.01), 2)
    PR_CURVE_DF_THRESHOLD = PR_CURVE_DF_THRESHOLD[['Standard_Threshold', 'Monotonic_Median_Precision', 'Monotonic_Median_Recall', 'Monotonic_Median_log2FC', 'F1_Score', 'Max_F1_Score']]
    
    # Save the threshold results to a CSV file and plot them.
    PR_CURVE_DF_THRESHOLD.to_csv(results_filename, sep = "\t", header = True, index = False)
    
    logging.info("Plot the validation statistics v. threshold values.")
    plot_threshold_calibration_stats(PR_CURVE_DF_THRESHOLD['Standard_Threshold'], PR_CURVE_DF_THRESHOLD['Monotonic_Median_Precision'], 
                                     PR_CURVE_DF_THRESHOLD['Monotonic_Median_Recall'], PR_CURVE_DF_THRESHOLD['Monotonic_Median_log2FC'],
                                     PR_CURVE_DF_THRESHOLD['F1_Score'], output_dir, args.prefix)
