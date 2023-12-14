python api_request_parallel_processor.py \
  --requests_filepath ../rlhf/gpt4-api2.jsonl \
  --save_filepath ../rlhf/gpt4-api-output2.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 5000 \
  --max_tokens_per_minute 300000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20