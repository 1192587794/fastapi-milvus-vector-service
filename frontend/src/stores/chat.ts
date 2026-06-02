import { create } from 'zustand'
import { SourceChunk } from '../api/chat'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceChunk[]
  confidence?: number
  timestamp: number
}

export interface ChatState {
  messages: Message[]
  sessionId: string | null
  loading: boolean
  addMessage: (message: Message) => void
  updateLastMessage: (content: string) => void
  appendToLastMessage: (content: string) => void
  setSessionId: (sessionId: string | null) => void
  setLoading: (loading: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  sessionId: null,
  loading: false,
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  updateLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages]
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content,
        }
      }
      return { messages }
    }),
  appendToLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages]
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content: messages[messages.length - 1].content + content,
        }
      }
      return { messages }
    }),
  setSessionId: (sessionId) => set({ sessionId }),
  setLoading: (loading) => set({ loading }),
  clearMessages: () => set({ messages: [], sessionId: null }),
}))
