import argparse
import json
import sys
from typing import *
import numpy as np

try:
    # import from notebook
    from .sectiontagger import SectionTagger
except ImportError as error:
    from sectiontagger import SectionTagger

section_tagger = SectionTagger()

from rouge_score import rouge_scorer
import pandas as pd
import string
import os


class DetailRouge:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def compute(self, references: list, predictions: list, **kwargs):
        results = [self.scorer.score(target, prediction) for target, prediction in zip(references, predictions)]
        df = []
        for each_result in results:
            rs = {}
            for r_type, v_type in each_result.items():
                rs[r_type + '-recall'] = v_type.recall
                rs[r_type + '-precision'] = v_type.precision
                rs[r_type + '-fmeasure'] = v_type.fmeasure
            df.append(rs)
        df = pd.DataFrame(df)
        return df.to_dict(orient='list')


SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

EMPTY_TAG = '#####EMPTY#####'


def detect_subsec(text: str):
    lines = text.split('\n')
    is_uppers = [line and line.isupper() for line in lines]
    for idx, line_cate in enumerate(is_uppers):
        previous_upper = [_idx for _idx, _line in enumerate(is_uppers[:idx]) if _line]
        if len(previous_upper) > 0:
            last_upper_idx = previous_upper[-1]
            last_upper_line = lines[last_upper_idx]
            previous_context = '\n'.join(lines[:last_upper_idx])
            if line_cate:
                content = '\n'.join(lines[last_upper_idx:idx])
                yield last_upper_line, len(previous_context), len(previous_context) + len(content)
            elif idx == len(lines) - 1:
                content = '\n'.join(lines[last_upper_idx:])
                yield last_upper_line, len(previous_context), len(previous_context) + len(content)


def kick_out_each_section_from_full_note(text: str, idx: int, non_kick_scores: dict) -> Tuple[List[Dict], list]:
    results = []
    to_kick = []
    non_kick_scores = {f'non_rm_score-{k}': v for k, v in non_kick_scores.items()}
    detected_divisions = section_tagger.divide_note_by_metasections(text)
    for detected_division in detected_divisions:
        label, _, _, start, _, end = detected_division
        kick_division_txt = text[:start] + text[end:]
        results.append({
            'sample_idx': idx,
            'rm': label.strip(),
            'rm_type': 'division',
            'text': kick_division_txt,
            **non_kick_scores
        })
        to_kick.append(label.strip())

    detected_sections = section_tagger.tag_sectionheaders(text)

    for detected_section in detected_sections:
        sectionheader, linenum, start, end = detected_section
        kick_header_txt = text[:start] + text[end:]
        results.append({
            'sample_idx': idx,
            'rm': sectionheader.strip(),
            'rm_type': 'section',
            'text': kick_header_txt,
            **non_kick_scores
        })
        to_kick.append(sectionheader.strip())

    detected_subsections = detect_subsec(text)

    for detected_section in detected_subsections:
        subsectionheader, start, end = detected_section
        kick_header_txt = text[:start] + text[end:]
        results.append({
            'sample_idx': idx,
            'rm': subsectionheader.strip(),
            'rm_type': 'header',
            'text': kick_header_txt,
            **non_kick_scores
        })
        to_kick.append(subsectionheader.strip())

    return results, to_kick


def kick_out_and_measure(full_note_preds: list, full_note_refs: list,
                         scorers: dict, non_rm_rs: dict = None) -> List[Dict]:
    print('Examining kick out every possible part....')
    to_measure = []

    if non_rm_rs is None:
        non_rm_rs = {}
        for name, (scorer, kwargs, keys, save_keys) in scorers.items():
            scores = scorer.compute(references=full_note_refs, predictions=full_note_preds, **kwargs)
            for score_key, save_key in zip(keys, save_keys):
                non_rm_rs[save_key] = scores[score_key]

    non_rm_rs = [(dict(zip(non_rm_rs.keys(), values))) for values in zip(*non_rm_rs.values())]

    predictions = []
    references = []
    for idx, (full_note, golden, non_kick_result) in enumerate(zip(full_note_preds, full_note_refs, non_rm_rs)):
        rm_results, _ = kick_out_each_section_from_full_note(full_note, idx, non_kick_result)
        to_measure.extend(rm_results)
        predictions.extend([d['text'] for d in rm_results])
        references.extend([golden, ] * len(rm_results))
    rm_rs = {}
    references = [x if x else 'none' for x in references]
    predictions = [x if x else 'none' for x in predictions]

    print(f'Re-scoring {len(references)} kicked samples ....')
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            rm_rs[save_key] = scores[score_key]
    rm_rs = {f'rm_score-{k}': v for k, v in rm_rs.items()}
    rm_rs = [(dict(zip(rm_rs.keys(), values))) for values in zip(*rm_rs.values())]
    results = []
    for sample_detail, sc_after_kick in zip(to_measure, rm_rs):
        sample_detail.update(sc_after_kick)
        results.append(sample_detail)
    return results


def print_detail_score(all_scores, cohorts, predictions, references, original_ids):
    results = []

    def _find_cohorts(idx, cohorts):
        for k, idxs in cohorts:
            if idx in idxs:
                yield k

    for idx, (pred, golden, o_idx) in enumerate(zip(predictions, references, original_ids)):
        sample_scores = {k: all_scores[k][idx] for k in all_scores.keys()}
        for score_cate in _find_cohorts(idx, cohorts):
            results.append({
                'sample-ID': o_idx,
                'score-cate': score_cate,
                'prediction': pred,
                'golden': golden,
                **sample_scores
            })
    return results


def gather_kickout_and_measure_decrease(rm_and_sc_results: List[Dict]) -> List[Dict]:
    print(f'Making reports for kicked samples...')
    df = pd.DataFrame(rm_and_sc_results)

    def get_avr_for_each_reduction(group: pd.DataFrame):
        score_types = [k.replace('non_rm_score-', '') for k in group.columns if 'non_rm_score-' in k]
        score_bf_kicked = {k: group[f'non_rm_score-{k}'].to_numpy() for k in score_types}
        score_af_kicked = {k: group[f'rm_score-{k}'].to_numpy() for k in score_types}

        score_reductions = {k: group[f'non_rm_score-{k}'].to_numpy() - group[f'rm_score-{k}'].to_numpy()
                            for k in score_types}
        non_zero_goldens = {k: np.where(group[f'non_rm_score-{k}'].to_numpy() == 0, 0.00001,
                                        group[f'non_rm_score-{k}'].to_numpy()) for k in score_types}
        score_reduction_percents = {k: score_reductions[k] / non_zero_goldens[k] for k in score_types}
        score_bf_kicked = {'full_note_score_bf_kicked-' + k: v.mean() for k, v in score_bf_kicked.items()}
        score_af_kicked = {'full_note_score_af_kicked-' + k: v.mean() for k, v in score_af_kicked.items()}
        score_reductions = {'full_note_score_reduction-' + k: v.mean() for k, v in score_reductions.items()}
        score_reduction_percents = {'full_note_score_reduction_percent-' + k: v.mean() * 100 for k, v in
                                    score_reduction_percents.items()}

        group_count = len(group)
        return [{'to-be-kicked': group.rm.tolist()[0],
                 'kick-type': group.rm_type.tolist()[0],
                 **score_bf_kicked,
                 **score_af_kicked,
                 **score_reductions,
                 **score_reduction_percents,
                 'count': group_count}]

    results = df.groupby('rm').apply(get_avr_for_each_reduction).explode().tolist()
    return results


def get_prefix_from_filepath(path):
    input_string = os.path.basename(path)
    input_string = input_string.split('.')[0]
    for punctuation in string.punctuation:
        if '_' != punctuation:
            input_string = input_string.replace(punctuation, '_')
    return input_string


def add_section_divisions(row, dialogue_column):
    for evaltype in ['reference', 'prediction']:
        text = row[evaltype]
        text_with_endlines = text.replace('__lf1__', '\n')
        detected_divisions = section_tagger.divide_note_by_metasections(text_with_endlines)
        for detected_division in detected_divisions:
            label, _, _, start, _, end = detected_division
            row['%s_%s' % (evaltype, label)] = text_with_endlines[start:end].replace('\n', '__lf1__')

    return row


def select_values_by_indices(lst, indices):
    return [lst[ind] for ind in indices]


def read_text(fn):
    with open(fn, 'r') as f:
        texts = f.readlines()
    return texts


def filter_and_aggregate(obj, indices):
    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='evaluate_summarization',
        description='This runs basic evaluation for both snippet (taskA) and full note summarization (taskB).'
    )
    parser.add_argument('--fn_gold', required=True, help='filename of gold references requires id and note column.')
    parser.add_argument('--fn_sys', required=True, help='filename of system references requires id and note column.')
    parser.add_argument(
        '--metadata_file', dest='fn_metadata', action='store', default=None,
        help='filename of metadata requires id and dataset column.'
    )
    parser.add_argument(
        '--task', action='store', default='taskB',
        help='summarization task, default is for full note (taskB). (use snippet, taskA, otherwise).'
    )
    parser.add_argument('--id_column', default='encounter_id', help='column to use for identifying id.')
    parser.add_argument('--note_column', default='note', help='column to use for identifying note.')
    parser.add_argument('--dialogue_column', default='dialogue', help='column to use for identifying dialogue.')
    parser.add_argument('--cate_column', default='section_header',
                        help='column to use for identifying category (for taskA only)')

    parser.add_argument(
        '--use_section_check', action='store_true', default=False,
        help='this flag with make sure evaluation shuts down for full task if 0 section divisions are detected.'
    )
    parser.add_argument(
        '--note_length_cutoff', default=512, type=int,
        help='Consider less than note_length_cutoff to be short and vice-versa for long'
    )
    parser.add_argument('--experiment', default='default', help='Prefix for save file.')
    parser.add_argument('-debug', default=False, action='store_true', help='If true, just runs eval over first example')
    parser.add_argument(
        '--detail_output_file',
        type=str, default='eval.xlsx',
        help='excel file path to save detailed evaluation.'
    )
    parser.add_argument(
        '--prefix',
        type=str, default='',
        help='sheet prefix for current validation section. For example: v5-train.'
    )
    parser.add_argument('--non_detail', action='store_true',
                        help='If true, run kick-out and save detail to excel file.')

    args = parser.parse_args()
    prefix = args.prefix
    if not prefix:
        prefix = get_prefix_from_filepath(args.fn_sys)

    # Read in reference/hyp files -added the latin encoding as one of the participants' file had a strange character somewhere
    # df_references = pd.read_csv(args.fn_gold, encoding='latin1')
    # df_predictions = pd.read_csv(args.fn_sys, encoding='latin1')
    df_references = pd.read_csv(args.fn_gold)
    df_predictions = pd.read_csv(args.fn_sys)

    print(f'Gold path: {args.fn_gold} ({len(df_references)} summaries)')
    print(f'System path: {args.fn_sys} ({len(df_predictions)} summaries)')


    def _conditional_rename(tmp_df, old_col, new_col):
        if new_col not in tmp_df.columns:
            tmp_df.rename(columns={old_col: new_col}, inplace=True)


    _conditional_rename(df_predictions, args.note_column, 'prediction')
    _conditional_rename(df_references, args.note_column, 'reference')
    # Only need id and prediction from df_predictions
    full_df = df_references.merge(df_predictions[[args.id_column, 'prediction']], on=args.id_column)
    full_df['dataset'] = 0

    original_ids = full_df[args.id_column].tolist()

    # create lists for references/predictions so we only need to calculate the scores once per instance
    references = full_df['reference'].tolist()
    predictions = full_df['prediction'].tolist()
    num_test = len(full_df)

    # =========== ADD SECTION DIVISIONS IF THIS IS THE FULL ENCOUNTER TASK ==========
    if args.task == 'taskB':

        for division in SECTION_DIVISIONS:
            references.extend(df_references[division].apply(lambda x: x.replace('\n', '__lf1__')))
            predictions.extend(df_predictions[division].apply(lambda x: x.replace('\n', '__lf1__')))
            original_ids.extend(full_df[args.id_column].tolist())

        # sanity check, we should now have 5 x the original set (one for full note, 4 for the divisions)
        rn = len(references)
        pn = len(predictions)
        en = len(full_df) * 5
        assert rn == pn == en, f'The number of references ({rn}) and predictions ({pn}) does not match expected ({en})'
    else:
        cates = None
        uniques = None
        if args.cate_column in df_references.columns:
            cates = df_references[args.cate_column].to_numpy()
            uniques = df_references[args.cate_column].unique().tolist()
        elif args.cate_column in df_predictions.columns:
            cates = df_predictions[args.cate_column].to_numpy()
            uniques = df_predictions[args.cate_column].unique().tolist()
    ######## Load Metrics from HuggingFace ########
    print('Loading ROUGE')
    import evaluate

    scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'rouge-detail': (
            DetailRouge(),
            {},
            ['rouge1-recall', 'rouge1-precision',
             'rouge1-fmeasure',
             'rouge2-recall',
             'rouge2-precision',
             'rouge2-fmeasure',
             'rougeL-recall',
             'rougeL-precision',
             'rougeL-fmeasure'],
            ['rouge1-recall', 'rouge1-precision',
             'rouge1-fmeasure',
             'rouge2-recall',
             'rouge2-precision',
             'rouge2-fmeasure',
             'rougeL-recall',
             'rougeL-precision',
             'rougeL-fmeasure']

        )
        #         'bert_scorer': (
        #             evaluate.load('bertscore', device='cpu'),
        #             {'model_type': 'microsoft/deberta-xlarge-mnli', 'device':'cpu'},
        #             ['precision', 'recall', 'f1'],
        #             ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        #         ),
        #         'bluert': (
        #             evaluate.load('bleurt', config_name='BLEURT-20'),
        #             {},
        #             ['scores'],
        #             ['bleurt']
        #         ),
    }

    ######## CALCULATE PER INSTANCE SCORES ########
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    cohorts = [
        ('all', list(range(num_test))),
    ]

    subsets = full_df['dataset'].unique().tolist()
    for subset in subsets:
        # Don't include anything after num_test (section-level)
        indices = full_df[full_df['dataset'] == subset].index.tolist()
        cohorts.append((f'dataset-{subset}', indices))

    if args.task == 'taskB':
        for ind, division in enumerate(SECTION_DIVISIONS):
            start = (ind + 1) * num_test
            end = (ind + 2) * num_test
            cohorts.append((f'division-{division}', list(range(start, end))))
    elif cates is not None:
        assert len(cates) == num_test
        for unique in uniques:
            idxs = np.argwhere(cates == unique).reshape(-1).tolist()
            cohorts.append((f'category-{unique}', idxs))

    outputs = {k: filter_and_aggregate(all_scores, idxs) for (k, idxs) in cohorts}

    # ###### OUTPUT TO JSON FILE ########
    fn_out = f'{args.experiment}_results.json'
    print(f'Saving results to {fn_out}')
    with open(fn_out, 'w') as fd:
        json.dump(outputs, fd, indent=4)

    for cohort, obj in outputs.items():
        print(cohort)
        for k, v in obj.items():
            print(f'\t{k} -> {round(v, 3)}')
        print('\n')

    if args.non_detail:
        print('Non-detailed results')
        sys.exit(0)

    # ###### OUTPUT TO EXCEL FILE ########
    fn_out = f'{args.detail_output_file}'
    overall = [{'cohort': cohort, **scs} for cohort, scs in outputs.items()]
    overall = pd.DataFrame(overall)

    details = print_detail_score(all_scores, cohorts, predictions, references, original_ids)
    details = pd.DataFrame(details)

    if args.task == 'taskB':
        full_note_preds = [predictions[idx] for idx in cohorts[0][1]]
        full_note_refs = [references[idx] for idx in cohorts[0][1]]
        non_rm_rs = {
            k: [all_scores[k][idx] for idx in cohorts[0][1]]
            for k in all_scores
        }
        kickout_details = kick_out_and_measure(full_note_preds=full_note_preds,
                                               full_note_refs=full_note_refs,
                                               scorers=scorers,
                                               non_rm_rs=non_rm_rs)
        kickout_reports = gather_kickout_and_measure_decrease(kickout_details)
        kickout_reports = pd.DataFrame(kickout_reports)
        if os.path.isfile(fn_out):
            with pd.ExcelWriter(fn_out, mode="a", if_sheet_exists="replace", ) as writer:
                overall.to_excel(writer, sheet_name=f'{prefix}-overall')
                details.to_excel(writer, sheet_name=f'{prefix}-detail')
                kickout_reports.to_excel(writer, sheet_name=f'{prefix}-kickout')
        else:
            with pd.ExcelWriter(fn_out, mode="w", ) as writer:
                overall.to_excel(writer, sheet_name=f'{prefix}-overall')
                details.to_excel(writer, sheet_name=f'{prefix}-detail')
                kickout_reports.to_excel(writer, sheet_name=f'{prefix}-kickout')
        print('Kick-out report')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(kickout_reports)
        print(f'Saving results to {fn_out}')
