# ============================================================
# Frontend Dockerfile
# ============================================================
# 多阶段构建：先构建，再用 Nginx 托管
# ============================================================

# 阶段 1：构建
FROM node:20-alpine as builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# 阶段 2：运行
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
