# -*- coding: utf-8 -*-
"""
Created on  06-03-2021
@author: Manjender Nir
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from multi_7Class_baseline import multi_class_c45_baseline_model_7class
# from mltat_multi_class_c45_improved_v1_7class_truncated_pkt_ml_pipeline import multi_class_c45_7class_truncated_pkt_ml_pipeline



def main():
	print(__doc__)
	print("Multi_class_C45 -> 7-class with truncated packet")
	multi_class_c45_baseline_model_7class()
	# multi_class_c45_7class_truncated_pkt_ml_pipeline()


if __name__ == "__main__":
	main()