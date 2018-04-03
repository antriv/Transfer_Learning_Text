import argparse
import json
import os
import pandas as pd

# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = "newsqa"
    target_dir = "newsqa_saved"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'test', out_name='test')
        prepro_each(args, 'train', out_name='train')
    elif args.mode == 'all':
        # create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'test', 0.0, 0.0, out_name='test')
        prepro_each(args, 'train', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, 1.0, out_name='train')
        prepro_each(args, 'dev', 0.0, 1.0, out_name='dev')
        prepro_each(args, 'test', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    print("Preprocessing data type %s" % data_type)
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}.csv".format(data_type))
    print("Reading data from source path %s" % source_path)
    source_data = pd.read_csv(source_path,
                           encoding='utf-8',
                           dtype=dict(is_answer_absent=float),
                           na_values=dict(question=[], story_text=[], validated_answers=[]),
                           keep_default_na=False)

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = [] # Gold standard answers
    span_answerss = [] # Answers from our spans
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data) * start_ratio))
    stop_ai = int(round(len(source_data) * stop_ratio))
    data_rows = source_data.iterrows()
    story_ids_to_idx = {}
    idx_to_story_ids = {}

    for ai, data_point in enumerate(tqdm(data_rows)):
        question_index, question_info = data_point[0], data_point[1]
        story_id = question_info['story_id']
        context = question_info['story_text']
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        question = question_info['question']
        question_id = ai
        answer_char_ranges = question_info['answer_char_ranges']

        # Copy get answer script from the newsqa dataset
        baseline_answers = []
        # Prefer validated answers.
        # If there are no validated answers, use the ones that are provided.
        if not 'validated_answers' in question_info or not question_info['validated_answers']:
            # Ignore per selection splits.
            char_ranges = question_info['answer_char_ranges'].replace('|', ',').split(',')
        else:
            validated_answers_dict = json.loads(question_info['validated_answers'])
            char_ranges = []
            for k, v in validated_answers_dict.items():
                char_ranges += v * [k]

        for char_range in char_ranges:
            if char_range.lower() == 'none':
               baseline_answers.append('NONE')
            elif ':' in char_range:
                start, end = map(int, char_range.split(':'))
                answer = question_info['story_text'][start:end]
                baseline_answers.append(answer)
        paragraph_ptr = -1
        pi = 0
        if story_id not in story_ids_to_idx:
            paragraph_ptr = len(story_ids_to_idx)
            story_ids_to_idx[story_id] = paragraph_ptr
            idx_to_story_ids[paragraph_ptr] = story_id  
            xp, cxp = [], []
            pp = []
            x.append(xp)
            cx.append(cxp)
            p.append(pp)

            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens

            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += 1
                    lower_word_counter[xijk.lower()] += 1
                    for xijkl in xijk:
                        char_counter[xijkl] += 1

        else:
            paragraph_ptr = story_ids_to_idx[story_id]
        rxi = [paragraph_ptr, pi]
        """
        print("TEST")
        print("TEST")
        print(story_ids_to_idx)
        print(len(xp))
        print(paragraph_ptr)
        """
        xi = x[paragraph_ptr][pi]

        qi = word_tokenize(question)
        cqi = [list(qij) for qij in qi]
        yi = []
        cyi = []
        answers = []
        answer_char_ranges_split = answer_char_ranges.split("|")
        for answer in answer_char_ranges_split:
            if answer == 'None':
                continue
            answer_char_range = answer.split(",")[0].split(":")
            answer_start = int(answer_char_range[0])
            answer_stop = int(answer_char_range[-1])
            answer_text = context[answer_start:answer_stop].strip()

            if answer_text == "":
                print("BAD ANSWER GIVEN %s" % answer_char_range)
                continue

            answers.append(answer_text)

            # TODO : put some function that gives word_start, word_stop here
            yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
            # yi0 = answer['answer_word_start'] or [0, 0]
            # yi1 = answer['answer_word_stop'] or [0, 1]


            assert len(xi[yi0[0]]) > yi0[1]
            assert len(xi[yi1[0]]) >= yi1[1]
            w0 = xi[yi0[0]][yi0[1]]
            w1 = xi[yi1[0]][yi1[1]-1]

            i0 = get_word_idx(context, xi, yi0)
            i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
            cyi0 = answer_start - i0
            cyi1 = answer_stop - i1 - 1

            #print(question, answer_text, w0[cyi0:], w1[:cyi1+1])
            #assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
            #assert answer_text[-1] == w1[-1]
            assert cyi0 < 32, (answer_text, w0)
            assert cyi1 < 32, (answer_text, w1)

            yi.append([yi0, yi1])
            cyi.append([cyi0, cyi1])

        for qij in qi:
            word_counter[qij] += 1
            lower_word_counter[qij.lower()] += 1
            for qijk in qij:
                char_counter[qijk] += 1

        q.append(qi)
        cq.append(cqi)
        y.append(yi)
        cy.append(cyi)
        rx.append(rxi)
        rcx.append(rxi)
        ids.append(question_id)
        idxs.append(len(idxs))
        answerss.append(baseline_answers)
        span_answerss.append(answers)
        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, 
            'span_answerss': span_answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p, 'story_ids_to_idx': story_ids_to_idx, 
              'idx_to_story_ids': idx_to_story_ids, 'word_counter': word_counter, 
              'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()