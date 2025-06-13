# Legal Compliance Guide for NewsXP AI Public Distribution

## Current Legal Status Analysis

**Your Use Case**: Non-commercial public MCP server + GitHub Pages for daily news aggregation

**Legal Risk Level**: 🟡 MEDIUM (with proper source filtering)

## ✅ SAFE SOURCES for Public Distribution

### Academic & Research (No Issues)
- `arxiv_ai`, `arxiv_ml`, `arxiv_cv` - Open access academic papers
- `google_research_blog` - Public research blog
- `microsoft_ai_blog` - Corporate research blog  
- `stanford_hai`, `mit_csail` - Academic institution news
- `pytorch_blog`, `tensorflow_blog` - Open source project updates
- `huggingface_papers_api` - Community platform

### Corporate Blogs (Generally Safe with Attribution)
- `openai_blog`, `anthropic_web`, `nvidia_blog` - Company announcements

## 🔴 RISKY SOURCES - DISABLE for Public Distribution

### Commercial News Outlets
- ❌ `mit_tech_review_ai` - Copyrighted journalism
- ❌ `techcrunch_ai` - Terms prohibit redistribution
- ❌ `venturebeat_ai` - Commercial content protection
- ❌ `import_ai_substack` - Newsletter with specific ToS

### Social Media Platforms  
- ❌ `reddit_ml` - Platform ToS restrictions
- ❌ `twitter_ai_search` - API terms prohibit redistribution
- ❌ `discord_ml_communities` - Community content restrictions

## 🛡️ LEGAL PROTECTION STRATEGIES

### 1. Content Limitations
```javascript
// Implement these limits in your code:
const LEGAL_LIMITS = {
  maxExcerptLength: 150,        // Fair use excerpt limit
  requireAttribution: true,      // Always show source
  linkToOriginal: true,         // Drive traffic back to source
  noFullArticles: true,         // Never redistribute complete articles
  respectRobotsTxt: true        // Honor crawling restrictions
};
```

### 2. Required Attribution Format
```html
<!-- Example attribution for each article -->
<article>
  <h3>Article Title (Brief Excerpt)</h3>
  <p>Brief 150-char excerpt...</p>
  <footer>
    Source: <a href="original_url">Original Article</a> 
    © 2025 Source Name - Fair Use Educational Purpose
  </footer>
</article>
```

### 3. Terms of Service for Your Site
```markdown
# Your Terms of Service Should Include:

1. **Educational/Research Purpose**: "Content aggregated for AI research and education"
2. **Fair Use Declaration**: "Brief excerpts used under fair use doctrine"
3. **Attribution Policy**: "All content attributed to original sources"
4. **DMCA Compliance**: "Takedown procedure for copyright holders"
5. **No Commercial Use**: "Service provided free for educational purposes"
```

## 📋 IMPLEMENTATION CHECKLIST

### Immediate Actions:
- [ ] Disable risky commercial news sources
- [ ] Implement 150-character excerpt limit
- [ ] Add proper attribution to all content
- [ ] Create Terms of Service for your site
- [ ] Add DMCA takedown procedure
- [ ] Ensure all links direct to original sources

### Code Changes Needed:
```python
# In your collect_news.py
def extract_safe_excerpt(content, max_length=150):
    """Extract safe fair-use excerpt"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."

def add_attribution(article):
    """Add required attribution"""
    article['attribution'] = {
        'source': article['source_name'],
        'url': article['original_url'],
        'copyright_notice': f"© {datetime.now().year} {article['source_name']}",
        'usage_type': 'fair_use_excerpt'
    }
    return article
```

## 🎯 FINAL RECOMMENDATION

**YES, it can be legal** with these conditions:

1. **Remove commercial news sources** (TechCrunch, MIT Tech Review, etc.)
2. **Keep only academic/research/open-source sources**
3. **Limit to brief excerpts** (150 chars max)
4. **Always link back to originals**
5. **Clear attribution on everything**
6. **Add proper Terms of Service**

**Estimated Legal Risk**: 🟢 LOW (if you follow above guidelines)

The key is focusing on academic/research sources and corporate blogs rather than commercial journalism. Your current source list has many safe options that would still provide valuable AI news coverage.

## Sources Recommended to Keep Enabled:
- ArXiv papers (all categories)
- Company research blogs (OpenAI, Google, Microsoft, etc.)
- Open source project updates (PyTorch, TensorFlow)
- Academic institutions (Stanford HAI, MIT CSAIL)
- Community platforms that allow redistribution (HuggingFace)

This approach gives you plenty of quality content while staying legally compliant for public distribution.
