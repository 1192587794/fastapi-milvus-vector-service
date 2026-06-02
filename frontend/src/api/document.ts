import api from './index'

export interface UpsertDocumentItem {
  id: string
  text: string
  source?: string
  tags?: string[]
  metadata?: Record<string, any>
}

export interface UpsertDocumentsRequest {
  items: UpsertDocumentItem[]
}

export interface UpsertDocumentsResponse {
  collection_name: string
  upserted_count: number
  primary_keys: string[]
}

export interface SearchDocumentsRequest {
  query_text: string
  top_k?: number
  source?: string
}

export interface SearchHit {
  id: string
  score: number
  text: string
  source?: string
  tags?: string[]
  metadata?: Record<string, any>
  created_at?: string
  updated_at?: string
}

export interface SearchDocumentsResponse {
  collection_name: string
  query_text: string
  top_k: number
  hits: SearchHit[]
}

export interface GetDocumentResponse {
  id: string
  text: string
  source?: string
  tags?: string[]
  metadata?: Record<string, any>
  created_at?: string
  updated_at?: string
}

export interface DeleteDocumentResponse {
  id: string
  deleted: boolean
}

export const upsertDocuments = (data: UpsertDocumentsRequest): Promise<UpsertDocumentsResponse> => {
  return api.post('/documents/upsert', data)
}

export const uploadDocument = async (
  file: File,
  source?: string,
  tags?: string[]
): Promise<UpsertDocumentsResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  if (source) formData.append('source', source)
  if (tags) formData.append('tags', tags.join(','))

  const response = await fetch('/api/v1/documents/upload', {
    method: 'POST',
    body: formData,
  })
  return response.json()
}

export const searchDocuments = (data: SearchDocumentsRequest): Promise<SearchDocumentsResponse> => {
  return api.post('/documents/search', data)
}

export const getDocument = (id: string): Promise<GetDocumentResponse> => {
  return api.get(`/documents/${id}`)
}

export const deleteDocument = (id: string): Promise<DeleteDocumentResponse> => {
  return api.delete(`/documents/${id}`)
}
