import re
import argparse
from tqdm import tqdm
from pathlib import Path


def main(args):
    with args.input.open('rU', encoding='iso-8859-1') as file_:
        cur_tag = None
        cur_ext_file = None
        for line in tqdm(file_):
            ext_match = re.match('<ext\Wn=(\d+)\W', line)
            if ext_match:
                cur_ext = ext_match.group(1) + '.txt'
                filepath = args.output / cur_ext
                cur_ext_file = filepath.open('w', encoding='utf-8')
                continue

            close_ext_match = re.match('</ext>', line)
            if close_ext_match:
                cur_ext_file.close()
                cur_ext_file = None
                continue

            tag_match = re.match('<([taps])>', line)
            if tag_match:
                cur_tag = tag_match.group(1)
                continue

            close_tag_match = re.match(f'</{cur_tag}>', line)
            if close_tag_match:
                cur_tag = None

            if cur_ext_file is not None:
                if cur_tag is not None:
                    cur_ext_file.write(line.strip())
                    cur_ext_file.write(' ')
                else:
                    cur_ext_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='CETEMPublico file in SGML format',
                        required=True, type=Path)
    parser.add_argument('--output', '-o',
                        help='Directory Path to which save the converted corpus',
                        required=True, type=Path)
    args = parser.parse_args()
    main(args)
