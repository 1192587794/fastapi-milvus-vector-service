import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, Space, Typography, Tag, Spin, message } from 'antd'
import { SendOutlined, ClearOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import { useChatStore } from '../stores/chat'
import { askQuestionStream, SourceChunk } from '../api/chat'

const { Text } = Typography

const ChatView: React.FC = () => {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, loading, addMessage, appendToLastMessage, setLoading, clearMessages, sessionId } = useChatStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: input.trim(),
      timestamp: Date.now(),
    }

    addMessage(userMessage)
    setInput('')
    setLoading(true)

    const assistantMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant' as const,
      content: '',
      sources: [],
      confidence: 0,
      timestamp: Date.now(),
    }
    addMessage(assistantMessage)

    try {
      const stream = askQuestionStream({
        question: userMessage.content,
        session_id: sessionId || undefined,
      })

      for await (const chunk of stream) {
        if (chunk.content) {
          appendToLastMessage(chunk.content)
        }
      }
    } catch (error) {
      console.error('Chat error:', error)
      message.error('对话失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const renderSources = (sources?: SourceChunk[]) => {
    if (!sources || sources.length === 0) return null

    return (
      <div style={{ marginTop: 12, padding: '8px 12px', background: '#f5f5f5', borderRadius: 8 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>参考来源：</Text>
        <div style={{ marginTop: 4 }}>
          {sources.map((source, index) => (
            <Tag key={source.id} color="blue" style={{ marginBottom: 4 }}>
              [{index + 1}] {source.source || '未知来源'} (相似度: {(source.score * 100).toFixed(1)}%)
            </Tag>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 180px)' }}>
      {/* 消息列表 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px 0' }}>
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '100px 0', color: '#999' }}>
            <RobotOutlined style={{ fontSize: 48, marginBottom: 16 }} />
            <div>开始提问吧！我可以回答基于文档的问题。</div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                marginBottom: 16,
                padding: '0 16px',
              }}
            >
              {msg.role === 'assistant' && (
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    background: '#1677ff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginRight: 12,
                    flexShrink: 0,
                  }}
                >
                  <RobotOutlined style={{ color: 'white', fontSize: 18 }} />
                </div>
              )}
              <div
                style={{
                  maxWidth: '70%',
                  padding: '12px 16px',
                  borderRadius: 12,
                  background: msg.role === 'user' ? '#1677ff' : '#f5f5f5',
                  color: msg.role === 'user' ? 'white' : '#333',
                }}
              >
                {msg.role === 'user' ? (
                  <div>{msg.content}</div>
                ) : (
                  <>
                    <div className="markdown-body">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    {renderSources(msg.sources)}
                  </>
                )}
              </div>
              {msg.role === 'user' && (
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    background: '#52c41a',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginLeft: 12,
                    flexShrink: 0,
                  }}
                >
                  <UserOutlined style={{ color: 'white', fontSize: 18 }} />
                </div>
              )}
            </div>
          ))
        )}
        {loading && messages[messages.length - 1]?.content === '' && (
          <div style={{ textAlign: 'center', padding: '16px 0' }}>
            <Spin tip="正在思考..." />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入框 */}
      <div style={{ padding: '16px 0', borderTop: '1px solid #f0f0f0' }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input.TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入问题... (Enter 发送，Shift+Enter 换行)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={loading}
            style={{ borderRadius: '8px 0 0 8px' }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={loading}
            style={{ borderRadius: '0 8px 8px 0', height: 'auto' }}
          />
        </Space.Compact>
        <div style={{ marginTop: 8, textAlign: 'right' }}>
          <Button
            icon={<ClearOutlined />}
            onClick={clearMessages}
            size="small"
            type="text"
          >
            清空对话
          </Button>
        </div>
      </div>
    </div>
  )
}

export default ChatView
