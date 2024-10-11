# Lab4SLC Swallow チュートリアル

```
# 作業ディレクトリに移動 (/path/to/dir は好きなところで良い、相対パスでもOK)
cd /path/to/dir

# 本チュートリアルのリポジトリをクローンして移動
git clone https://github.com/nara-wu-slc/llm-jp_tutorial.git
cd llm-jp_tutorial

# 本チュートリアル用のPython仮想環境を /path/to/dir/llm-jp_tutorial/.venv に作成
python3 -m venv .venv

# 仮想環境を有効化
source .venv/bin/activate
```

```
# vLLM をインストール (Pytorch等もまとめてインストールされる》
pip3 install vllm
```

```
# モデル保存用のキャッシュディレクトリの設定を読み込む
source /slc/share/dot.zshrc.slc
```

```
# ChatGPT的な対話を行うデモプログラムを回す
# CUDA_VISIBLE_DEVICES=0 は1枚目のGPUだけを使う、というおまじない
# -m の後はモデルの名前 (https://huggingface.co/collections/tokyotech-llm/llama-31-swallow-66fd4f7da32705cadd1d5bc6 参照、但し 70B はメモリに載らないので使えない)
# "-Instruct" がついたモデルを使う想定
# プログラムの立ち上げが終わると
# input>
# というプロンプトが出てくるので文字列を入力。空行のままEnterすると入力終了（入力が空であればプログラムを閉じる）
env CUDA_VISIBLE_DEVICES=0 ./sample_interactive.py -m Llama-3.1-Swallow-8B-Instruct-v0.1
```

実行例
```
INFO 10-11 09:03:09 llm_engine.py:226] Initializing an LLM engine (v0.6.1.dev238+ge2c6e0a82) with config: model='tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1', speculative_config=None, tokenizer='tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1, use_v2_block_manager=False, num_scheduler_steps=1, multi_step_stream_outputs=False, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
INFO 10-11 09:03:11 model_runner.py:1014] Starting to load model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1...
INFO 10-11 09:03:12 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:42<02:07, 42.34s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [01:24<01:24, 42.18s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [02:07<00:42, 42.45s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [02:17<00:00, 29.70s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [02:17<00:00, 34.33s/it]

INFO 10-11 09:05:30 model_runner.py:1025] Loading model weights took 14.9595 GB
INFO 10-11 09:05:32 gpu_executor.py:122] # GPU blocks: 13330, # CPU blocks: 2048
INFO 10-11 09:05:34 model_runner.py:1329] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-11 09:05:34 model_runner.py:1333] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-11 09:05:43 model_runner.py:1456] Graph capturing finished in 9 secs.
input>ドラえもんの兄について教えてください
input>
ドラえもんの兄はドラえもんの兄であるドラミです。
input>
```
