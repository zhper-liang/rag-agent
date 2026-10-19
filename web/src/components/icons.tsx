// 线条图标：JSX 内联 SVG，不走 innerHTML，CSP 下无需额外放行。
import type { ReactNode } from 'react';

function Icon({ children, size = 18 }: { children: ReactNode; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}

export const IconPlus = () => <Icon><circle cx="12" cy="12" r="9" /><path d="M12 8v8M8 12h8" /></Icon>;
export const IconSearch = () => <Icon><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></Icon>;
export const IconSidebar = () => <Icon><rect x="3" y="4" width="18" height="16" rx="3" /><path d="M9 4v16" /></Icon>;
export const IconMore = () => <Icon><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></Icon>;
export const IconEdit = () => <Icon><path d="M4 20h4L19 9l-4-4L4 16v4Z" /><path d="m13.5 6.5 4 4" /></Icon>;
export const IconPin = () => <Icon><path d="M9 4h6l-1 5 3 3v2H7v-2l3-3-1-5Z" /><path d="M12 14v6" /></Icon>;
export const IconTrash = () => <Icon><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></Icon>;
export const IconArrowUp = () => <Icon size={20}><path d="M12 19V5M6 11l6-6 6 6" /></Icon>;
export const IconStop = () => <Icon size={16}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" /></Icon>;
export const IconCopy = () => <Icon size={16}><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></Icon>;
export const IconRetry = () => <Icon size={16}><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5" /></Icon>;
export const IconThumbUp = () => <Icon size={16}><path d="M7 11v9H4v-9h3Zm0 0 4-7a2 2 0 0 1 2 2v4h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 20H7" /></Icon>;
export const IconThumbDown = () => <Icon size={16}><path d="M7 13V4H4v9h3Zm0 0 4 7a2 2 0 0 0 2-2v-4h5a2 2 0 0 0 2-2.3l-1.2-6A2 2 0 0 0 16.8 4H7" /></Icon>;
export const IconFilter = () => <Icon size={16}><path d="M4 5h16l-6 8v6l-4-2v-4L4 5Z" /></Icon>;
export const IconClose = () => <Icon><path d="M6 6l12 12M18 6 6 18" /></Icon>;
export const IconDoc = () => <Icon size={16}><path d="M6 3h8l4 4v14H6V3Z" /><path d="M14 3v4h4M9 13h6M9 17h6" /></Icon>;
export const IconExternal = () => <Icon size={16}><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></Icon>;
export const IconDownload = () => <Icon size={16}><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></Icon>;
export const IconLogo = () => (
  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
    <rect x="2" y="2" width="20" height="20" rx="6" fill="var(--accent)" />
    <path d="M7 8h6.5a3 3 0 0 1 0 6H10l4 3M7 8v9" stroke="#fff" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const IconAttach = () => <Icon><path d="m21 11-8 8a6 6 0 0 1-8.5-8.5l9-9a4 4 0 0 1 5.7 5.7l-9 9a2 2 0 0 1-2.8-2.8l8.5-8.5" /></Icon>;
