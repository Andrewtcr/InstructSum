python api_request_parallel_processor.py \
  --requests_filepath ../data/articles.jsonl \
  --save_filepath ../data/summaries.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 1000 \
  --max_tokens_per_minute 10000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20