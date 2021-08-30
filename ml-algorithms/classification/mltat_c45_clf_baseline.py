
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from dataPreparation.featureEngineering.transformation import PreprocessorPipelineFactory

FEATURES_ALL = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb', 'min_itf', 'mean_itf',
                'mx_itf', 'sdv_itf','min_itb', 'mean_itb', 'mx_itb', 'sdv_itb', 'tot_dur', 'prot', 'num_pf', 'tot_bf',
                'num_pb', 'tot_bb', 'tot_plf', 'tot_plb', 'fst_plsf', 'fst_plsb', 'num_plpf', 'num_plpb', 'ent_pb',
                'ent_itb', 'actual_label']
FEATURES = ['min_pf', 'mean_pf', 'mx_pf', 'sdv_pf', 'min_pb', 'mean_pb', 'mx_pb', 'sdv_pb', 'min_itf', 'mean_itf',
            'mx_itf', 'sdv_itf', 'min_itb', 'mean_itb', 'mx_itb', 'sdv_itb', 'tot_dur', 'prot', 'num_pf', 'tot_bf',
            'num_pb', 'tot_bb', 'tot_plf', 'tot_plb', 'fst_plsf', 'fst_plsb', 'num_plpf', 'num_plpb', 'ent_pb',
            'ent_itb']


def multi_class_c45_baseline_model_7class():
    print("hello ")
    dict_2020a_res = {}
    dict_2020a = {
        'audio_chat_messenger': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\audio_chat\messenger",
    # 	'audio_chat_skype': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\audio_chat\skype",
    # 	'audio_stream_sound_cloud': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\audio_stream\sound-cloud",
    # 	'audio_stream_spotify': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\audio_stream\spotify",
    # 	'file_transfer_dropbox_download': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\file_transfer\dropbox\download",
    # 	'file_trnsfer_dropbox_upload': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\file_transfer\dropbox\upload",
    # 	'file_transfer_google_drive_download': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\file_transfer\google-drive\download",
    # 	'video_stream__netfilx': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\video_stream\netflix",
    # 	'video_stream_youtube': r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a-2020a-merged-15files\video_stream\youtube"
    }
    dict_2020a_d = {
        0: r'C:\Users\Owner\mltat\data\mltat\Dataset_External\for_manjinder_7_classes'
        # 1: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\audio_chat\messenger",
        # 2: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\audio_chat\skype",
        # 3: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\audio_stream\sound-cloud",
        # 4: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\audio_stream\spotify",
        # 5: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\file_transfer\dropbox\download",
        # 6: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\file_transfer\dropbox\upload",
        # 7: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\file_transfer\google-drive\download",
        # 8: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\mail\gmail",
        # 9: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\mail\solana-mail",
        # 10: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\text_chat\messenger",
        # 11: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\text_chat\telegram",
        # 12: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\video_chat\messenger",
        # 13: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\video_chat\skype",
        # 14: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\video_stream\netflix",
        # 15: r"C:\Users\Owner\projects3\Generalization\tasks\task-2\data\Solana2020a_2020d_merged-35files\video_stream\youtube"
    }
    dict_gen_2020a = {
	    1: r"C:\Users\Owner\projects3\Generalization\tasks\task-4\task-4a\mltat-external-data\data",
        # 2: r"C:\Users\Owner\mltat\data\mltat\Dataset_External\for_manjinder_7_classes"
    }
    conf_res = ''

    for k, v in dict_gen_2020a.items():
        files = listFiles([v])
        print(files[0])
        print(files[1])
        df_train, df_test = read_train_test_from_files(train_file=files[0], test_file=files[1])
        print(df_train)
        print(df_test)
        conf_res, multi_class_counts = cv_train(df_train, df_test)
        dict_2020a_res[k] = [conf_res, multi_class_counts]
    print("====== dictionary_confusion_matrix ======")
    print(dict_2020a_res)

    # ====================================== 7 Class ==================================================
    columns = ['audio', 'audio_chat', 'file_transfer', 'network_service', 'p2p', 'social_media', 'video']
    df_conf = pd.DataFrame(conf_res, columns=columns)
    new_columns = ['audio', 'audio_chat', 'network_service', 'file_transfer', 'video', 'p2p', 'social_media']
    df_conf_reorder = df_conf.reindex(columns=new_columns)
    print(df_conf_reorder)
    df_conf_reorder.to_csv(
        r'C:\Users\Owner\projects3\Generalization\tasks\task3\task-3a\mltat-external-data\task-3a-1-conf_matrix_log-minmax-from ml.csv',
        index=False)


def read_train_test_from_files(train_file=None, test_file=None):
    df_test = pd.read_csv(test_file, names=FEATURES_ALL)
    df_train = pd.read_csv(train_file, names=FEATURES_ALL)
    return df_train, df_test


def getConfusioMatrix(true_labels=None, predicted_labels=None):
    conf_res = confusion_matrix(true_labels, predicted_labels)
    multi_class_labels, multi_class_counts = np.unique(true_labels, return_counts=True)
    print(multi_class_labels)
    print(conf_res)
    print(multi_class_counts)
    return conf_res, multi_class_counts


def cv_train(df_train, df_test):
    print("\nTraining starts: ")

    random_seed = 3572203868
    np.random.seed(random_seed)
    X = df_train[FEATURES]
    y = df_train['actual_label'].copy()
    y_arr  = y.values
    df_to_test = df_test[FEATURES].astype(np.float32)
    true_labels = df_test['actual_label'].values.tolist()
    print("====== Class Distribution ========")
    print(df_test.groupby(df_test.columns[-1]).size())
    class_names = df_train['actual_label'].unique()
    transform = False
    # procname = 'log'
    # procname = 'log_minmax'
    # procname = 'log_stdscaler'
    # procname = 'stdscaler'
    procname = 'minmax'
    arr_X = ''
    if transform:
        print(PreprocessorPipelineFactory() \
            .get_pipeline(procname=procname))
        arr_X = PreprocessorPipelineFactory() \
            .get_pipeline(procname=procname) \
            .fit_transform(X[FEATURES])
        X = pd.DataFrame(data=arr_X, columns=FEATURES)
    print(arr_X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # use pipeline to do classification
    clf_dt1 = DecisionTreeClassifier(random_state=42)
    clf_dt1 = clf_dt1.fit(X_train, y_train)

    # ===== Evaluate clf_dt1 ==========
    print("========= clf_dt1 ===============")
    predicted_labels = clf_dt1.predict(df_to_test)
    conf_res, multi_class_counts = getConfusioMatrix(true_labels=true_labels, predicted_labels=predicted_labels)

    # plot the tree
    plt.figure(figsize=(15, 7.5))
    plot_tree(clf_dt1, filled=True, rounded=True, class_names=class_names, feature_names=X.columns)
    # plt.show()

    """":parameter
    Cost Complexity Pruning Part 1: Visualize alpha
    """

    path = clf_dt1.cost_complexity_pruning_path(X_train, y_train)  # determine the value for alpha
    ccp_alphas = path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]
    print("+===== clf_dt1.ccp_alphas =======")
    print(ccp_alphas)
    print("+===== clf_dt1.ccp_alphas count =======")
    print(len(set(ccp_alphas)))
    #
    clf_dts = []
    count = 1
    for ccs_alpha in set(ccp_alphas):
        print('{}: {}'.format(count, ccs_alpha))
        count += 1
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccs_alpha)
        clf_dt.fit(X_train, y_train)
        clf_dts.append(clf_dt)

    for clf_dt in clf_dts:
        print("========= clf_dt1 ===============")
        predicted_labels = clf_dt.predict(df_to_test)
        getConfusioMatrix(true_labels=true_labels, predicted_labels=predicted_labels)

    train_scores = [clf_dt1.score(X_train, y_train) for clf_dt in clf_dts]
    test_Scores = [clf_dt1.score(X_test, y_test) for clf_dt in clf_dts]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alphas for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_Scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    #
    # """
    #     Cost Complexity pruning Part 2: Cross Validation for finding the Best Alpha
    # """
    # clf_dt_cv = DecisionTreeClassifier(random_state=42, ccp_alpha=0, )
    # scores = cross_val_score(clf_dt_cv, X_train, y_train, cv=5)
    # df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
    # df.plot(x='tree', y='accuracy', title='clf_dt_cv', marker='o', linestyle='--')
    # # plt.show()
    #
    # # cross_val = True
    # # if cross_val:
    # alpha_loop_values = []
    # for ccp_alpha in ccp_alphas:
    #     clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    #     scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    #     alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    #
    # alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
    # alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
    # # plt.show()
    # print("===== alpha_results cross_validation =====")
    # print(alpha_results)
    # # ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014) & (alpha_results['alpha'] < 0.015)]['alpha']
    # ideal_ccp_alpha = alpha_results.iloc[alpha_results['mean_accuracy'].idxmax()]['alpha']
    # print("============ ideal_ccp_alpha ============")
    # print(ideal_ccp_alpha)
    #
    # """:parameter
    #     Building, Evaluating, Drawing, and Interpreting the Final Classification Tree
    # """
    # clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
    # clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
    # print("============ clf_dt_pruned ============")
    # predicted_labels = clf_dt_pruned.predict(df_to_test)
    # clf_dt_pruned.tree_.
    # conf_res, multi_class_counts = getConfusioMatrix(true_labels=true_labels, predicted_labels=predicted_labels)
    # # disp = plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels=class_names)
    # # disp.ax_.set_title("Pruned Confuaion matrix")
    # # print(disp.confusion_matrix)
    #
    # # Draw pruned tree
    # plt.figure(figsize=(15, 7.5))
    # plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=class_names, feature_names=X.columns)
    # # plt.show()
    # # ======== Calculating Feature Importance ================
    # feat_importance = clf_dt_pruned.tree_.compute_feature_importances(normalize=False)
    # feat_imp_dict = dict(zip(X.columns, clf_dt_pruned.feature_importances_))
    # feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
    # feat_imp.rename(columns={0: 'FeatureImportance'}, inplace=True)
    # feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head()

    return conf_res, multi_class_counts


def listFiles(paths=None):
    files = []
    list_paths = []
    # r=root, d=directories, f = files
    # print(paths)
    for path in paths:
        for p, d, f in os.walk(path):
            list_paths.append(p)
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(p, file))
    # files.append(file)
    return files


def main():
    multi_class_c45_baseline_model_7class()


if __name__ == "__main__":
    main()

