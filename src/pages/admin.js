import React, { useEffect } from 'react';

export default function AdminPage() {
  useEffect(() => {
    // 重定向到真正的 admin 界面
    if (typeof window !== 'undefined') {
      window.location.href = '/admin/';
    }
  }, []);

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h2>正在跳转到内容管理系统...</h2>
        <p>如果没有自动跳转，请点击 <a href="/admin/">这里</a></p>
      </div>
    </div>
  );
} 