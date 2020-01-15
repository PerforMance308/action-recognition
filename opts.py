import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='./data/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='ucf101',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default='annotation/ucf101_01.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='ucf101',
        type=str,
        help='Used dataset (ucf101 | hmdb51)')
    parser.add_argument(
        '--n_classes',
        default=101,
        type=int,
        help=
        'Number of classes (ucf101: 101, hmdb51: 51)'
    )
    
    args = parser.parse_args()

    return args
