// History API 路由：/、/c/{session_id}、/d/{doc_id}?s={section}。后端对这三类路径都返回 index.html。
import { useEffect, useState } from 'react';
import { isDocId } from './render';
import { isSessionId } from './sessions';

export type Route =
  | { name: 'home' }
  | { name: 'chat'; sessionId: string }
  | { name: 'doc'; docId: string; section: string };

export function parseRoute(pathname: string, search = ''): Route {
  const chat = /^\/c\/([^/]+)\/?$/.exec(pathname);
  if (chat && isSessionId(chat[1])) return { name: 'chat', sessionId: chat[1]! };
  const doc = /^\/d\/([^/]+)\/?$/.exec(pathname);
  if (doc && isDocId(doc[1])) return { name: 'doc', docId: doc[1]!, section: new URLSearchParams(search).get('s') ?? '' };
  return { name: 'home' };
}

export function routePath(route: Route): string {
  if (route.name === 'chat') return `/c/${route.sessionId}`;
  if (route.name === 'doc') return `/d/${route.docId}${route.section ? `?s=${encodeURIComponent(route.section)}` : ''}`;
  return '/';
}

// state.back：从页内打开原文时记下来源，关闭抽屉走 history.back()，直接打开链接时没有它。
export function navigate(route: Route, { replace = false, back = false } = {}): void {
  const path = routePath(route);
  if (path === location.pathname + location.search) return;
  const state = back ? { back: true } : null;
  if (replace) history.replaceState(state, '', path);
  else history.pushState(state, '', path);
  window.dispatchEvent(new PopStateEvent('popstate', { state }));
}

export function canGoBack(): boolean {
  return Boolean((history.state as { back?: boolean } | null)?.back);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(location.pathname, location.search));
  useEffect(() => {
    const on = () => setRoute(parseRoute(location.pathname, location.search));
    window.addEventListener('popstate', on);
    return () => window.removeEventListener('popstate', on);
  }, []);
  return route;
}
