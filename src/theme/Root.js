import React from 'react';
import NetlifyIdentity from '../components/NetlifyIdentity';

// 这个组件包装整个应用
export default function Root({children}) {
  return (
    <>
      <NetlifyIdentity />
      {children}
    </>
  );
} 