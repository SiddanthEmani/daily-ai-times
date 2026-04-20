# AI Model Usage Statistics

| Model | Max Context Length | Type | Quota |
|-------|-------------------|------|-------|
| **qwen-3-235b-a22b-instruct-2507** | 64,000 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 60,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **qwen-3-235b-a22b-thinking-2507** | 65,536 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 60,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **qwen-3-coder-480b** | 65,536 | Requests | minute: 10<br>hour: 100<br>day: 100 |
| | | Tokens | minute: 150,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **llama-3.3-70b** | 65,536 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 64,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **qwen-3-32b** | 65,536 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 64,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **llama3.1-8b** | 8,192 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 60,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **llama-4-scout-17b-16e-instruct** | 8,192 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 60,000<br>hour: 1,000,000<br>day: 1,000,000 |
| **llama-4-maverick-17b-128e-instruct** | 8,192 | Requests | minute: 30<br>hour: 900<br>day: 14,400 |
| | | Tokens | minute: 60,000<br>hour: 1,000,000<br>day: 1,000,000 |

## Key Observations

- **Context lengths vary significantly**: 8,192 to 65,536 tokens
- **Most restrictive quota**: qwen-3-coder-480b has only 10 requests/minute, 100 requests/hour and day
- **Highest token quota**: qwen-3-coder-480b allows 150,000 tokens/minute