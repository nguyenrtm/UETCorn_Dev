# ImageCLEF MEDIQA-Sum-2023 

Website: https://www.imageclef.org/2023/medical/mediqa

Google group: https://groups.google.com/g/mediqa-nlp


# Tasks

The MEDIQA-Sum task focuses on the automatic summarization and classification of doctor-patient conversations through three subtasks:

* Subtask A - Dialogue2Topic Classification.  Given a conversation snippet between a doctor and patient, participants are tasked with identifying the topic (associated section header). Topics/Section headers will be one of twenty normalized common section labels (e.g. Assessment, Diagnosis, Exam, Medications, Past Medical History).

* Subtask B - Dialogue2Note Summarization. Given a conversation snippet between a doctor and patient and a section header, participants are tasked with producing a clinical note section text summarizing the conversation.

* Subtask C - Full-Encounter Dialogue2Note Summarization. Given a full encounter conversation between a doctor and patient, participants are tasked with producing a full clinical note summarizing the conversation.


# Submission Guidelines

* Each team is allowed to submit a maximum of 3 runs for each task.

* Each task submission will consist of two parts:
  1. An online form based submission of run outputs. (Please use the appropriate link below).
     - Task B: https://forms.gle/H3TfKQZs5qGk32JC9
     - Task C: https://forms.gle/5tC2QtJxvjXXxz3h7
  3. Code submission - teams will be asked to share their github repo with the organizers along with their output submissions.
     - Specifically, please add the following usernames: abachaa, griff4692, wyim

* Output formats:
  - The output should be in the *.csv format and should be named task{A|B|C}_teamName_run{1|2|3}_mediqaSum.csv
  - Task A/B:
      - The run files should be a csv file with 2 columns: "TestID", "SystemOutput"
      - For task A this should be your topic label, for task B it will be your snippet summary
  - Task C:
      - The run files should be a csv file with at least 6 columns: "encounter_id", "note", "subjective", "objective_exam", "objective_results", "assessment_and_plan"
      - "note" should be the full clinical note, equivalent to the concatenation of "subjective", "objective_exam", "objective_results", "assessment_and_plan"

* Please see below for code organization guidelines. We will provide further details along with the release of the test sets on May 3 (TaskA), May 8 (TaskB/C), 2023.

* After the competition, we encourage the teams to release their codes publicly with the publication of their papers.


## Code Submission Guidelines

Please make sure that your inference code:

(1) can run within 48 hours for a size comparable to the train set,
(2) takes less than the resources needed for a Standard_NC24 machines from azure ( https://learn.microsoft.com/en-us/azure/virtual-machines/nc-series ),
(3) can run in a linux environment.

In your main code repository, please create subfolders for each subtask that you are participating in.
For each subtask, your code package should at least include the following three files:

**install.sh**
- create an environment {teamname}_task{A,B,C}_venv
- activate your environment
- install all your requirements
- then close your environment

**activate.sh**
- activate your environment

**decode_task{A,B,C}_run{1,2,3}.sh**
- first argument should be the source text
- it is expected that the order of the resulting output be the same as the source text

In summary, for us to run we should be able to get quick-started by doing the following commands:

```
./install.sh
./activate.sh
decode_task{A,B,C}_run{1,2,3}.sh [input-csv-file]
```
- The input should be in the *.csv format of the original test files
- The output files should have the same format as the runs and should be named task{A|B|C}_teamName_run{1|2|3}_mediqaSum.csv 


 * Our copy of the codes and models will be deleted after the release of the official results and confirmation emails will be sent to the participants. 

## Evaluation

- For the three tasks, we will use ensemble metrics that correlate well with human judgments. 
- These ensemble metrics combine SOTA evaluation metrics including ROUGE, BERTScore and BLEURT. We will provide further details along with the release of the test sets on May 3 (TaskA), May 8 (TaskB/C), 2023.
(Please use the evaluate_summarization_v2.py as a guide)
- Organisers will evaluate the submissions and release the results to the participants on May 17.
- Results will be considered official after submitting a working notes paper on June 5, 2023.

# Schedule

* 20 March 2023: Release of the training and validation sets
* 3 May 2023: Release of the test set for subtask A
* 5 May 2023: Run submission deadline for subtask A
* 8 May 2023: Release of the test sets for subtasks B & C (with Ground truth of subtask A)
* 10 May 2023: Run submission deadline for subtasks B & C
* 17 May 2023: Release of the processed results by the task organizers
* 5 June 2023: Deadline for submitting working notes papers by the participants
* 23 June 2023: Notification of acceptance of the working notes papers
* 7 July 2023: Camera ready copy of participant papers
* 18-21 September 2023: CLEF 2023, Thessaloniki, Greece


# Organizers

* Wen-wai Yim, Microsoft, USA
* Asma Ben Abacha, Microsoft, USA
* Neal Snider, Microsoft/Nuance, USA
* Griffin Adams, Columbia University, USA
* Meliha Yetisgen, University of Washington, USA
