import argparse
import sys
DATA_DIR = ''


def mk_extract(E):
    pass


def external_data_parse(T):
    pass


def external_clf_train(df_one):
    pass


def external_data_read(all_train_data_file):
    pass


def external_clf_validate(df_one):
    pass


def main():
    parser = argparse.ArgumentParser(description="A script to trigger external classifier")
    parser.add_argument('-T', metavar='data_dir', nargs='?', const=DATA_DIR, \
                        help='Train external classifier')
    parser.add_argument('-E', metavar="data_dir", nargs='?', const=DATA_DIR, \
                        help='Extract features from PCAPs found at data_dir')
    parser.add_argument('-V', metavar='data_dir', nargs='?', const=DATA_DIR, \
                        help='Run cross validation for trained model')
    args = parser.parse_args()

    if (not args.E and not args.T and not args.V) or (args.E and args.T and args.V):
        print("Please specify one option: -E, -T or -V.\n")
        parser.print_help()
        sys.exit(1)

    if args.E:
        mk_extract(args.E)
    elif args.T:
        df_one = external_data_parse(args.T)
        external_clf_train(df_one)
    elif args.V:
        df_one = external_data_read(all_train_data_file)
        external_clf_validate(df_one)


if __name__=="__main__":
    main()