import api from './index'

export interface AskRequest {
  question: string
  top_k?: number
  source?: string
  temperature?: number
  max_tokens?: number
  session_id?: string
}

export interface SourceChunk {
  id: string
  text: string
  score: number
  source?: string
  metadata?: Record<string, any>
}

export interface AskResponse {
  question: string
  answer: string
  sources: SourceChunk[]
  llm_provider: string
  confidence: number
  hybrid_recall_used: boolean
  graph_recall_used: boolean
  query_rewrite_used: boolean
  session_id?: string
}

export const askQuestion = (data: AskRequest): Promise<AskResponse> => {
  return api.post('/qa/ask', data)
}

export const askQuestionStream = async function* (
  data: AskRequest
): AsyncGenerator<{ content?: string; done?: boolean }> {
  const response = await fetch('/api/v1/qa/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('ReadableStream not supported')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (data === '[DONE]') {
          return
        }
        try {
          yield JSON.parse(data)
        } catch (e) {
          console.warn('Failed to parse SSE data:', data)
        }
      }
    }
  }
}
