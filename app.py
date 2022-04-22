import pandas as pd
from lstm import MODEL

if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    LSTM = MODEL()
    train_x, train_y = LSTM.training_process(args.training)
    model, offset = LSTM.model_training(train_x, train_y)
    test_x = LSTM.testing_process(args.testing, args.training)
    action = LSTM.make_pred(test_x, offset, model)
    
    result = pd.DataFrame()
    result['action'] = action
    result.to_csv(args.output,index=0,header=None)
