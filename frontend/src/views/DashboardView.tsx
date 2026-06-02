import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Tag, Descriptions, Spin, message } from 'antd'
import {
  ClusterOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  ApiOutlined,
} from '@ant-design/icons'
import { getGraphStats, GraphStatsResponse } from '../api/graph'

const DashboardView: React.FC = () => {
  const [stats, setStats] = useState<GraphStatsResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    setLoading(true)
    try {
      const data = await getGraphStats()
      setStats(data)
    } catch (error) {
      console.error('Failed to load stats:', error)
      message.error('加载统计信息失败')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <Spin size="large" tip="加载中..." />
      </div>
    )
  }

  return (
    <div>
      {/* 概览卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="实体总数"
              value={stats?.total_entities || 0}
              prefix={<ClusterOutlined />}
              valueStyle={{ color: '#1677ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="关系总数"
              value={stats?.total_relations || 0}
              prefix={<NodeIndexOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="文档数量"
              value={stats?.documents_count || 0}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="实体类型"
              value={stats ? Object.keys(stats.entity_type_counts).length : 0}
              prefix={<ApiOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 实体类型分布 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="实体类型分布">
            {stats?.entity_type_counts && Object.keys(stats.entity_type_counts).length > 0 ? (
              <Descriptions column={1}>
                {Object.entries(stats.entity_type_counts).map(([type, count]) => (
                  <Descriptions.Item key={type} label={type}>
                    <Tag color="blue">{count}</Tag>
                  </Descriptions.Item>
                ))}
              </Descriptions>
            ) : (
              <div style={{ textAlign: 'center', color: '#999', padding: 40 }}>
                暂无数据
              </div>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="关系类型分布">
            {stats?.relation_type_counts && Object.keys(stats.relation_type_counts).length > 0 ? (
              <Descriptions column={1}>
                {Object.entries(stats.relation_type_counts).map(([type, count]) => (
                  <Descriptions.Item key={type} label={type}>
                    <Tag color="green">{count}</Tag>
                  </Descriptions.Item>
                ))}
              </Descriptions>
            ) : (
              <div style={{ textAlign: 'center', color: '#999', padding: 40 }}>
                暂无数据
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 系统信息 */}
      <Card title="系统信息">
        <Descriptions column={2}>
          <Descriptions.Item label="向量数据库">Milvus</Descriptions.Item>
          <Descriptions.Item label="Embedding 模型">Ollama nomic-embed-text</Descriptions.Item>
          <Descriptions.Item label="LLM 模型">Ollama qwen2.5:7b</Descriptions.Item>
          <Descriptions.Item label="图存储后端">NetworkX</Descriptions.Item>
          <Descriptions.Item label="会话存储">Redis</Descriptions.Item>
          <Descriptions.Item label="框架">FastAPI + React</Descriptions.Item>
        </Descriptions>
      </Card>
    </div>
  )
}

export default DashboardView
