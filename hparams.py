import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # origin segment files
    parser.add_argument('--train_en', default='./data/train/train.en',
                             help="english training segmented data")
    parser.add_argument('--train_fr', default='./data/train/train.fr',
                             help="french training segmented data")
    parser.add_argument('--test_en', default='./data/test/newstest2013.en',
                             help="english evaluation segmented data")
    parser.add_argument('--test_fr', default='./data/test/newstest2013.fr',
                             help="french evaluation segmented data")

    # generate middle ids file
    parser.add_argument('--train_en_ids', default='./data/train/train_ids.en',
                             help="english training ids data")
    parser.add_argument('--train_fr_ids', default='./data/train/train_ids.fr',
                             help="french training ids data")
    parser.add_argument('--test_en_ids', default='./data/test/newstest2013_ids.en',
                             help="english evaluation ids data")
    parser.add_argument('--test_fr_ids', default='./data/test/newstest2013_ids.fr',
                             help="french evaluation ids data")

    # vocabulary_size
    parser.add_argument('--en_vocab_size', default=32000, type=int)
    parser.add_argument('--fr_vocab_size', default=32000, type=int)

    # vocabulary files
    parser.add_argument('--vocab_en', default='./data/vocab/vocab.en',
                        help="english vocabulary file path")
    parser.add_argument('--vocab_fr', default='./data/vocab/vocab.fr',
                        help="french vocabulary file path")

    # training scheme
    parser.add_argument('--learning_rate', default=0.5, type=float,help='learning_rate')
    parser.add_argument('--learning_rate_decay_factor', default=0.99, type=float, help='Learning rate decays by this much.')
    parser.add_argument('--max_gradient_norm', default=5.0, type=float, help='Clip gradients to this norm.')


    parser.add_argument('--batch_size', default=64, type=int,help='Batch size to use during training.')
    parser.add_argument('--size', default=30, type=int,help='Size of each model layer.')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of layers in the model.')
    parser.add_argument('--max_train_data_size', default=10000, type=int,help='Limit on the size of training data (0: no limit).')
    parser.add_argument('--steps_per_checkpoint', default=200, type=int,help='How many training steps to do per checkpoint.')
    parser.add_argument('--steps', default=20, type=int,
                        help='train steps')

    # training chekcpoint file
    parser.add_argument('--train_checkpoint_dir', default='./train_checkpoint',
                             help="train checkpoint dir")

    # train/inference
    parser.add_argument('--decode', default=False, type=bool,help='Set to True for interactive decoding.')
    parser.add_argument('--self_test', default=False, type=bool,help='Run a self-test if this is set to True.')

