python rec.py --user-min 0 --user-max 100 --topk 500 --model ItemCF --sim cos --alpha 0.15 --output rec_result.txt
python evaluation.py rec_result.txt
