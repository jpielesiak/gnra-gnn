opens test folder and finds how many times a given example was marked positive despite being negative
getFalsePositiveFromTest.py

using the output of that creates a small csv with only interestingly overvoted candidates
uv run filter_candidates.py falsePositives.csv FPtoAdd.csv models_test/approximate-affinity-propagation-decision-tree.csv --threshold 16

creates mmcifs with them
 uv run addFPtoTrainSet.py -csv ./FPtoAdd.csv \
        -mirror ./test_cif \
        -out candidate_cif_files