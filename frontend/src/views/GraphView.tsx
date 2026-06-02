import React, { useState, useEffect, useRef } from 'react'
import { Card, Input, Button, Space, Select, Tag, Spin, message, Descriptions, Empty, Typography } from 'antd'
import { SearchOutlined, ReloadOutlined, ClusterOutlined } from '@ant-design/icons'
import * as echarts from 'echarts'
import { useGraphStore } from '../stores/graph'
import { getSubgraph, getGraphStats, SubgraphNode } from '../api/graph'

const { Search } = Input
const { Option } = Select
const { Text } = Typography

const entityColors: Record<string, string> = {
  Disease: '#ff4d4f',
  Symptom: '#faad14',
  Drug: '#52c41a',
  Procedure: '#1677ff',
  Department: '#722ed1',
  AnatomicalPart: '#13c2c2',
  MedicalDevice: '#eb2f96',
  Other: '#8c8c8c',
}

const GraphView: React.FC = () => {
  const [searchValue, setSearchValue] = useState('')
  const [depth, setDepth] = useState(1)
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)
  const { nodes, edges, stats, loading, setGraphData, setStats, setLoading, setSelectedNode } = useGraphStore()

  useEffect(() => {
    loadStats()
    loadGraph()
    return () => {
      chartInstance.current?.dispose()
    }
  }, [])

  useEffect(() => {
    if (chartRef.current && nodes.length > 0) {
      renderChart()
    }
  }, [nodes, edges])

  const loadStats = async () => {
    try {
      const data = await getGraphStats()
      setStats(data)
    } catch (error) {
      console.error('Failed to load graph stats:', error)
    }
  }

  const loadGraph = async (entityName?: string) => {
    setLoading(true)
    try {
      const data = await getSubgraph({
        entity_name: entityName || undefined,
        depth: depth,
      })
      setGraphData(data.nodes, data.edges)
      if (data.nodes.length === 0) {
        message.info('未找到相关实体')
      }
    } catch (error) {
      console.error('Failed to load graph:', error)
      message.error('加载图谱失败')
    } finally {
      setLoading(false)
    }
  }

  const renderChart = () => {
    if (!chartRef.current) return

    if (chartInstance.current) {
      chartInstance.current.dispose()
    }

    const chart = echarts.init(chartRef.current)
    chartInstance.current = chart

    const categories = Array.from(new Set(nodes.map((n) => n.type))).map((type) => ({
      name: type,
    }))

    const option: echarts.EChartsOption = {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.dataType === 'node') {
            const node = params.data
            return `
              <div>
                <strong>${node.name}</strong><br/>
                类型: ${node.type}<br/>
                ${node.attributes ? Object.entries(node.attributes).map(([k, v]) => `${k}: ${v}`).join('<br/>') : ''}
              </div>
            `
          }
          if (params.dataType === 'edge') {
            return `${params.data.source} --(${params.data.relation_type})--> ${params.data.target}`
          }
          return ''
        },
      },
      legend: {
        data: categories.map((c) => c.name),
        orient: 'vertical',
        right: 10,
        top: 20,
      },
      series: [
        {
          type: 'graph',
          layout: 'force',
          data: nodes.map((node) => ({
            id: node.id,
            name: node.name,
            type: node.type,
            attributes: node.attributes,
            symbolSize: 30,
            category: categories.findIndex((c) => c.name === node.type),
            itemStyle: {
              color: entityColors[node.type] || '#8c8c8c',
            },
            label: {
              show: true,
              position: 'right',
              formatter: '{b}',
            },
          })),
          links: edges.map((edge) => ({
            source: edge.source,
            target: edge.target,
            relation_type: edge.relation_type,
            lineStyle: {
              width: Math.max(1, edge.confidence * 3),
              curveness: 0.3,
            },
            label: {
              show: true,
              formatter: edge.relation_type,
              fontSize: 10,
            },
          })),
          categories: categories,
          force: {
            repulsion: 200,
            gravity: 0.1,
            edgeLength: 150,
            layoutAnimation: true,
          },
          roam: true,
          draggable: true,
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 4,
            },
          },
        },
      ],
    }

    chart.setOption(option)

    chart.on('click', (params: any) => {
      if (params.dataType === 'node') {
        const node = params.data as SubgraphNode
        setSelectedNode(node)
        setSearchValue(node.name)
        loadGraph(node.name)
      }
    })

    window.addEventListener('resize', () => chart.resize())
  }

  const handleSearch = (value: string) => {
    if (value.trim()) {
      loadGraph(value.trim())
    }
  }

  const handleRefresh = () => {
    loadStats()
    loadGraph(searchValue || undefined)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 180px)' }}>
      {/* 统计信息 */}
      {stats && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Descriptions size="small" column={4}>
            <Descriptions.Item label="实体总数">
              <Tag color="blue">{stats.total_entities}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="关系总数">
              <Tag color="green">{stats.total_relations}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="文档数量">
              <Tag color="orange">{stats.documents_count}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="实体类型">
              <Tag color="purple">{Object.keys(stats.entity_type_counts).length}</Tag>
            </Descriptions.Item>
          </Descriptions>
        </Card>
      )}

      {/* 搜索栏 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <Search
            placeholder="输入实体名称搜索"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            onSearch={handleSearch}
            style={{ width: 300 }}
            enterButton={<SearchOutlined />}
          />
          <Select value={depth} onChange={setDepth} style={{ width: 120 }}>
            <Option value={1}>1 跳</Option>
            <Option value={2}>2 跳</Option>
            <Option value={3}>3 跳</Option>
          </Select>
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
            刷新
          </Button>
          <Button onClick={() => loadGraph()}>
            全图
          </Button>
        </Space>
      </Card>

      {/* 图谱可视化 */}
      <Card
        style={{ flex: 1 }}
        bodyStyle={{ padding: 0, height: '100%' }}
      >
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Spin size="large" tip="加载中..." />
          </div>
        ) : nodes.length === 0 ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Empty
              image={<ClusterOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
              description="暂无图谱数据，请先上传文档并构建图谱"
            />
          </div>
        ) : (
          <div ref={chartRef} style={{ width: '100%', height: '100%' }} />
        )}
      </Card>

      {/* 图例说明 */}
      <Card size="small" style={{ marginTop: 16 }}>
        <Space wrap>
          <Text type="secondary">实体类型：</Text>
          {Object.entries(entityColors).map(([type, color]) => (
            <Tag key={type} color={color}>
              {type}
            </Tag>
          ))}
        </Space>
      </Card>
    </div>
  )
}

export default GraphView
