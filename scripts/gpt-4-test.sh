python api_request_parallel_processor.py \
  --requests_filepath gpt-4-test-prompts.jsonl \
  --save_filepath gpt-4-test-output.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 5000 \
  --max_tokens_per_minute 300000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20