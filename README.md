# justice-served
NLP Project: binary classification of criticality for Swiss legal cases (i.e. likelihood of a case being referred to the supreme court)


Here are the dependencies:

```bash
conda install -c pytorch pytorch cpuonly
pip install -r adapter_script/requirements.txt
pip install -U adapter-transformers
```

I've been trying to run like this:

```bash
cd adapter_script
./run.sh --model_name="Musixmatch/umberto-commoncrawl-cased-v1" --type="hierearchical" --language="it" --train_language="it" --mode="train" --sub_datasets=False --seed=123 --debug=True
```