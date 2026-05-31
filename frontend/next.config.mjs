/** @type {import('next').NextConfig} */
const nextConfig = {
  // FastAPI backend runs on port 5000
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:5000/:path*",
      },
    ];
  },
};

export default nextConfig;
