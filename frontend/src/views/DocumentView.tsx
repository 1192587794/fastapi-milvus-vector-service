import React, { useState } from 'react'
import { Card, Upload, Button, Table, Space, Tag, message, Popconfirm, Input } from 'antd'
import { UploadOutlined, DeleteOutlined, SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import { uploadDocument, searchDocuments, deleteDocument, SearchHit } from '../api/document'

const { Dragger } = Upload

const DocumentView: React.FC = () => {
  const [searchText, setSearchText] = useState('')
  const [searchResults, setSearchResults] = useState<SearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  const handleUpload = async (file: UploadFile) => {
    setUploading(true)
    try {
      const result = await uploadDocument(file as any, 'upload')
      message.success(`上传成功，处理了 ${result.upserted_count} 个分片`)
    } catch (error) {
      console.error('Upload error:', error)
      message.error('上传失败')
    } finally {
      setUploading(false)
    }
    return false
  }

  const handleSearch = async () => {
    if (!searchText.trim()) {
      message.warning('请输入搜索内容')
      return
    }
    setLoading(true)
    try {
      const result = await searchDocuments({ query_text: searchText, top_k: 10 })
      setSearchResults(result.hits)
    } catch (error) {
      console.error('Search error:', error)
      message.error('搜索失败')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteDocument(id)
      message.success('删除成功')
      setSearchResults(searchResults.filter((item) => item.id !== id))
    } catch (error) {
      console.error('Delete error:', error)
      message.error('删除失败')
    }
  }

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 200,
      ellipsis: true,
    },
    {
      title: '内容',
      dataIndex: 'text',
      key: 'text',
      ellipsis: true,
      render: (text: string) => (
        <div style={{ maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {text}
        </div>
      ),
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 100,
      render: (source: string) => <Tag color="blue">{source || '未知'}</Tag>,
    },
    {
      title: '相似度',
      dataIndex: 'score',
      key: 'score',
      width: 100,
      render: (score: number) => `${(score * 100).toFixed(1)}%`,
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      width: 150,
      render: (tags: string[]) => (
        <Space>
          {tags?.map((tag) => (
            <Tag key={tag}>{tag}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (_: any, record: SearchHit) => (
        <Popconfirm
          title="确定删除这个文档吗？"
          onConfirm={() => handleDelete(record.id)}
          okText="确定"
          cancelText="取消"
        >
          <Button type="link" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Popconfirm>
      ),
    },
  ]

  return (
    <div>
      {/* 上传区域 */}
      <Card title="上传文档" style={{ marginBottom: 16 }}>
        <Dragger
          name="file"
          multiple={false}
          accept=".pdf,.docx"
          beforeUpload={handleUpload}
          showUploadList={false}
          disabled={uploading}
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">支持 PDF、DOCX 格式文件</p>
        </Dragger>
      </Card>

      {/* 搜索区域 */}
      <Card title="文档搜索">
        <Space style={{ marginBottom: 16 }}>
          <Input.Search
            placeholder="输入关键词搜索文档"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onSearch={handleSearch}
            style={{ width: 400 }}
            enterButton={<SearchOutlined />}
            loading={loading}
          />
          <Button icon={<ReloadOutlined />} onClick={() => setSearchResults([])}>
            清空
          </Button>
        </Space>

        <Table
          columns={columns}
          dataSource={searchResults}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: '暂无搜索结果' }}
        />
      </Card>
    </div>
  )
}

export default DocumentView
