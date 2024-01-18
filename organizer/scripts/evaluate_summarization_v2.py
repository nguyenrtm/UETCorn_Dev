import sys
import json
import argparse

import evaluate
import pandas as pd
import numpy as np

from sectiontagger import SectionTagger
section_tagger = SectionTagger()


SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

EMPTY_TAG = '#####EMPTY#####'

def add_section_divisions(row, dialogue_column ):
    for evaltype in ['reference', 'prediction']:
        text = row[evaltype]
        text_with_endlines = text.replace( '__lf1__', '\n' )
        detected_divisions = section_tagger.divide_note_by_metasections(text_with_endlines)
        for detected_division in detected_divisions:
            label, _, _, start, _, end = detected_division
            row[ '%s_%s' % (evaltype, label)] = text_with_endlines[start:end].replace('\n', '__lf1__')

    return row


def select_values_by_indices(lst, indices) :
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


if __name__ == "__main__" :
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

    args = parser.parse_args()

    # Read in reference/hyp files -added the latin encoding as one of the participants' file had a strange character somewhere
    #df_references = pd.read_csv(args.fn_gold, encoding='latin1')
    #df_predictions = pd.read_csv(args.fn_sys, encoding='latin1')
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

    # create lists for references/predictions so we only need to calculate the scores once per instance
    references = full_df['reference'].tolist()
    predictions = full_df['prediction'].tolist()
    num_test = len(full_df)

    # =========== ADD SECTION DIVISIONS IF THIS IS THE FULL ENCOUNTER TASK ==========
    if args.task == 'taskB' :

        for division in SECTION_DIVISIONS:
            references.extend( df_references[ division ].apply( lambda x: x.replace('\n','__lf1__' ) ) )
            predictions.extend( df_predictions[ division ].apply( lambda x: x.replace('\n','__lf1__' ) ) )

        # sanity check, we should now have 5 x the original set (one for full note, 4 for the divisions)
        rn = len(references)
        pn = len(predictions)
        en = len(full_df) * 5
        assert rn == pn == en, f'The number of references ({rn}) and predictions ({pn}) does not match expected ({en})'

    ######## Load Metrics from HuggingFace ########
    print('Loading ROUGE, BERTScore, BLEURT from HuggingFace')
    scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore', device='cpu'),
            {'model_type': 'microsoft/deberta-xlarge-mnli', 'device':'cpu'},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
        'bluert': (
            evaluate.load('bleurt', config_name='BLEURT-20'),
            {},
            ['scores'],
            ['bleurt']
        ),
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
