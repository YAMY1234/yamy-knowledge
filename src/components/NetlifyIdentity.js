import { useEffect } from 'react';

export default function NetlifyIdentity() {
  useEffect(() => {
    if (typeof window !== 'undefined' && window.netlifyIdentity) {
      // 初始化 Netlify Identity
      window.netlifyIdentity.on('init', user => {
        if (!user) {
          // 监听登录和注册事件
          window.netlifyIdentity.on('login', () => {
            window.location.href = '/admin/';
          });
          
          window.netlifyIdentity.on('signup', () => {
            window.location.href = '/admin/';
          });
        }
      });

      // 处理邀请链接和密码重置
      window.netlifyIdentity.on('ready', () => {
        const urlParams = new URLSearchParams(window.location.search);
        
        // 检查是否是邀请或密码重置链接
        if (urlParams.get('invitation_token') || urlParams.get('recovery_token')) {
          console.log('检测到邀请或密码重置链接，打开认证界面');
          window.netlifyIdentity.open();
        }
      });

      // 监听关闭事件
      window.netlifyIdentity.on('close', () => {
        console.log('认证界面已关闭');
      });
    }
  }, []);

  return null; // 这个组件不渲染任何内容
} 