import { create } from 'zustand'
import { SubgraphNode, SubgraphEdge, GraphStatsResponse } from '../api/graph'

export interface GraphState {
  nodes: SubgraphNode[]
  edges: SubgraphEdge[]
  stats: GraphStatsResponse | null
  loading: boolean
  selectedNode: SubgraphNode | null
  setNodes: (nodes: SubgraphNode[]) => void
  setEdges: (edges: SubgraphEdge[]) => void
  setGraphData: (nodes: SubgraphNode[], edges: SubgraphEdge[]) => void
  setStats: (stats: GraphStatsResponse) => void
  setLoading: (loading: boolean) => void
  setSelectedNode: (node: SubgraphNode | null) => void
  clearGraph: () => void
}

export const useGraphStore = create<GraphState>((set) => ({
  nodes: [],
  edges: [],
  stats: null,
  loading: false,
  selectedNode: null,
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  setGraphData: (nodes, edges) => set({ nodes, edges }),
  setStats: (stats) => set({ stats }),
  setLoading: (loading) => set({ loading }),
  setSelectedNode: (node) => set({ selectedNode: node }),
  clearGraph: () => set({ nodes: [], edges: [], selectedNode: null }),
}))
