# Adım 1: Statik dosyaları oluşturmak (React/Vue gibi bir framework olsaydı burada build adımı olurdu)
# Vanilla JS kullandığımız için bu adıma gerek yok, direkt Nginx kullanabiliriz.

# Adım 2: Nginx sunucusu ile statik dosyaları sun
FROM nginx:1.21-alpine

# `dist` klasöründeki web sitesi dosyalarını Nginx'in HTML klasörüne kopyala
COPY . /usr/share/nginx/html

# Port 80'i dışarıya aç
EXPOSE 80

# Nginx'i başlat
CMD ["nginx", "-g", "daemon off;"]