import api from './index'

export interface Entity {
  id: string
  name: string
  type: string
  attributes?: Record<string, any>
  doc_id: string
  chunk_id?: string
}

export interface Relation {
  source_id: string
  target_id: string
  relation_type: string
  confidence: number
  doc_id: string
  chunk_id?: string
}

export interface GraphBuildRequest {
  doc_id: string
  text: string
}

export interface GraphBuildResponse {
  doc_id: string
  entities_count: number
  relations_count: number
}

export interface GraphQueryRequest {
  query: string
  max_hops?: number
  top_k?: number
}

export interface GraphQueryResponse {
  query: string
  entities: Entity[]
  relations: Relation[]
  source_chunks: string[]
}

export interface SubgraphRequest {
  entity_name?: string
  depth?: number
}

export interface SubgraphNode {
  id: string
  name: string
  type: string
  attributes?: Record<string, any>
}

export interface SubgraphEdge {
  source: string
  target: string
  relation_type: string
  confidence: number
}

export interface SubgraphResponse {
  nodes: SubgraphNode[]
  edges: SubgraphEdge[]
}

export interface GraphStatsResponse {
  total_entities: number
  total_relations: number
  entity_type_counts: Record<string, number>
  relation_type_counts: Record<string, number>
  documents_count: number
}

export interface GraphDeleteResponse {
  doc_id: string
  deleted_entities: number
  deleted_relations: number
}

export const buildGraph = (data: GraphBuildRequest): Promise<GraphBuildResponse> => {
  return api.post('/graph/build', data)
}

export const getGraphStats = (): Promise<GraphStatsResponse> => {
  return api.get('/graph/stats')
}

export const queryGraph = (data: GraphQueryRequest): Promise<GraphQueryResponse> => {
  return api.post('/graph/query', data)
}

export const getSubgraph = (data: SubgraphRequest): Promise<SubgraphResponse> => {
  return api.post('/graph/subgraph', data)
}

export const deleteGraph = (docId: string): Promise<GraphDeleteResponse> => {
  return api.delete(`/graph/${docId}`)
}
