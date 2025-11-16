# DeepTranscribe - Advanced Features Documentation

## Vector RAG (Retrieval-Augmented Generation) System

DeepTranscribe now features a sophisticated Vector RAG system that enables intelligent retrieval of relevant information from your transcripts and chat history.

### How RAG Works

1. **Vector Embeddings**: All transcripts and chat messages are converted into vector embeddings using Google's Gemini embedding model
2. **Similarity Search**: When you ask a question, the system searches for semantically similar content using cosine similarity
3. **Context Injection**: Relevant content is injected into the chatbot's context for accurate, informed responses
4. **User-Specific**: All retrieval is scoped to the logged-in user's data only

### RAG Features

- **Semantic Search**: Find relevant information even when exact keywords don't match
- **Top-K Retrieval**: Retrieves the top 5 most relevant segments by default
- **Configurable Threshold**: Minimum similarity score of 0.3 ensures quality matches
- **Multi-Source**: Searches both transcript segments and chat history

## Advanced Memory System

The chatbot features four different memory types to suit different conversation needs:

### 1. Summary Buffer Memory (Recommended - Default)

**Best for**: Most conversations requiring both context and recent detail

**How it works**:
- Maintains a summary of older conversations
- Keeps recent 5 messages in full detail
- Provides perfect balance of context and recency

**Use when**:
- You want comprehensive understanding with recent context
- Having long conversations that build on previous topics
- Need the bot to remember key decisions and facts

### 2. Buffer Memory

**Best for**: Quick questions, fast responses

**How it works**:
- Stores only the most recent 10 messages
- No summarization overhead
- Fastest response time

**Use when**:
- Asking quick, independent questions
- Don't need extensive conversation history
- Want the fastest possible responses

### 3. Summary Memory

**Best for**: Long-term conversation understanding

**How it works**:
- Generates comprehensive summaries of all conversations
- Includes conversations from the last 7 days
- No specific recent messages included

**Use when**:
- Want to understand overall conversation themes
- Need big-picture understanding
- Asking about trends or patterns

### 4. Graph Memory

**Best for**: Complex relationship tracking

**How it works**:
- Extracts entities (people, places, organizations, concepts)
- Builds knowledge graph of relationships
- Tracks connections across conversations

**Use when**:
- Tracking multiple related entities
- Understanding complex relationships
- Need to recall who said what about whom

## Enhanced User Dashboard

Access your comprehensive dashboard at `/api/dashboard/`

### Features

1. **Summary Statistics**
   - Total transcripts created
   - Total duration (hours and minutes)
   - Total segments transcribed
   - Chat message counts (user vs assistant)

2. **Visual Analytics**
   - Transcripts Over Time chart (last 30 days)
   - Top 10 Speaker Activity bar chart
   - Interactive visualizations powered by Chart.js

3. **Recent Transcripts Table**
   - Quick view of your latest transcripts
   - Status indicators (Complete/In Progress)
   - Direct links to both standard and enhanced views

## Enhanced Transcript Detail View

Access the enhanced view at `/api/transcripts/{id}/enhanced/`

### Features

1. **Advanced Statistics**
   - Duration, segment count, total words
   - Average confidence score
   - Embeddings count

2. **Speaker Distribution Chart**
   - Visual representation of speaker participation
   - Segment counts per speaker

3. **Minute-by-Minute Breakdown**
   - Detailed view of each minute in the transcript
   - Speaker activity per minute
   - Word count and duration per minute
   - All segments grouped by minute

## API Endpoints

### Analytics APIs

```
GET /api/dashboard/
```
Returns comprehensive user analytics dashboard

```
GET /api/transcripts/list/?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&search=query
```
Get filtered list of transcripts with search and date filters

```
GET /api/transcripts/{id}/analytics/
```
Get detailed analytics for a specific transcript including:
- Speaker timeline
- Top 20 words used
- Speaker statistics
- Word frequency analysis

```
POST /api/chatbot/
Body: {
  "message": "Your question",
  "memory_type": "summary_buffer"  // optional, defaults to summary_buffer
}
```
Interact with the chatbot using selected memory type

### Memory Types

Available memory type values:
- `summary_buffer` (recommended)
- `buffer`
- `summary`
- `graph`

## Usage Examples

### Using Different Memory Types

1. **For General Conversation** (use Summary Buffer):
```javascript
{
  "message": "What did we discuss about the project last week?",
  "memory_type": "summary_buffer"
}
```

2. **For Quick Questions** (use Buffer):
```javascript
{
  "message": "What's the weather?",
  "memory_type": "buffer"
}
```

3. **For Relationship Tracking** (use Graph):
```javascript
{
  "message": "Who worked with Sarah on the design project?",
  "memory_type": "graph"
}
```

### RAG-Powered Queries

The chatbot automatically uses RAG to find relevant transcript segments:

```
User: "Tell me about the meeting with John"
Bot: [Searches all transcripts for mentions of John, retrieves relevant segments, and provides informed response]
```

```
User: "What decisions were made about the budget?"
Bot: [Retrieves all budget-related transcript segments and summarizes decisions]
```

## Configuration

### Memory System Settings

Edit `api/memory_manager.py` to customize:

```python
# Buffer size for Buffer Memory
BufferMemory(user, k=10)  # Change k to adjust number of messages

# Summary threshold for Summary Buffer
SummaryBufferMemory(user, buffer_size=5, summary_threshold=20)

# Days to include in Summary Memory
week_ago = datetime.now() - timedelta(days=7)  # Change days as needed
```

### RAG Settings

Edit `api/chatbot_service.py` to customize:

```python
# Minimum similarity threshold for RAG retrieval
self.retrieval_threshold = 0.3  # Lower = more results, higher = more precise

# Number of results to retrieve
relevant_chats, relevant_transcripts = self.get_relevant_content(query, top_k=5)
```

## Best Practices

1. **Memory Type Selection**
   - Start with Summary Buffer for most use cases
   - Switch to Buffer for quick, unrelated questions
   - Use Graph when tracking complex relationships
   - Use Summary for understanding overall themes

2. **RAG Optimization**
   - Be specific in your questions for better retrieval
   - Ask about topics, not just keywords
   - The system understands semantic meaning, not just exact matches

3. **Dashboard Usage**
   - Check dashboard regularly for usage insights
   - Use charts to identify conversation patterns
   - Track speaker participation across transcripts

## Performance Considerations

- **Embeddings**: Generated once per message/segment and cached
- **Memory Loading**: Lazy-loaded only when needed
- **Chart Rendering**: Client-side using Chart.js
- **RAG Search**: Efficient cosine similarity using NumPy

## Troubleshooting

### Slow Response Times

If responses are slow:
1. Switch to Buffer memory for faster responses
2. Check if embeddings need to be regenerated
3. Verify Gemini API key is valid

### Inaccurate Responses

If responses don't seem accurate:
1. Try Graph memory for better entity tracking
2. Lower the RAG threshold for more context
3. Use Summary memory for better overall understanding

### Missing Context

If bot seems to forget things:
1. Switch to Summary Buffer (includes more context)
2. Check that embeddings are being generated
3. Verify memory system is properly initialized

## Future Enhancements

Planned features:
- PostgreSQL with pgvector for faster similarity search
- Redis caching for embeddings
- Background task processing for embedding generation
- Advanced graph visualizations
- Export analytics reports
- Custom memory configurations per user
