python api_request_parallel_processor.py \
  --requests_filepath ../data/gpt-3.5-test-prompts.jsonl \
  --save_filepath ../data/gpt-3.5-test-output.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 3500 \
  --max_tokens_per_minute 160000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20