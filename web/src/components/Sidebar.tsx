// 侧栏：新对话、会话搜索、按日期分组的会话列表，每项有 … 菜单（重命名 / 置顶 / 删除）。
import { useEffect, useRef, useState } from 'react';
import { groupSessions } from '../lib/sessions';
import type { Session } from '../lib/types';
import { IconEdit, IconLogo, IconMore, IconPin, IconPlus, IconSearch, IconSidebar, IconTrash } from './icons';

interface Props {
  sessions: Session[];
  currentId: string;
  onNew: () => void;
  onOpen: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onPin: (id: string, pinned: boolean) => void;
  onDelete: (id: string) => void;
  onCollapse: () => void;
}

export function Sidebar({ sessions, currentId, onNew, onOpen, onRename, onPin, onDelete, onCollapse }: Props) {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [menu, setMenu] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const groups = groupSessions(sessions, query);

  return (
    <nav className="side" aria-label="会话">
      <div className="side-head">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); onNew(); }}>
          <IconLogo /><span>RAG 问答</span>
        </a>
        <div className="side-tools">
          <button type="button" className="icon-btn" aria-label="搜索会话" aria-pressed={searching}
            onClick={() => { setSearching(!searching); setQuery(''); }}><IconSearch /></button>
          <button type="button" className="icon-btn" aria-label="收起侧栏" onClick={onCollapse}><IconSidebar /></button>
        </div>
      </div>
      <button type="button" className="new-chat" onClick={onNew}><IconPlus /> 开启新对话</button>
      {searching && (
        <input className="side-search" type="search" placeholder="搜索会话标题" aria-label="搜索会话标题" autoFocus
          value={query} onChange={(e) => setQuery(e.target.value)} />
      )}
      <div className="side-list">
        {!groups.length && <p className="side-empty">{query ? '没有匹配的会话' : '还没有会话'}</p>}
        {groups.map((g) => (
          <section key={g.label} aria-label={g.label}>
            <h2 className="group-label">{g.label}</h2>
            <ul>
              {g.sessions.map((s) => (
                <SessionItem key={s.id} session={s} current={s.id === currentId}
                  menuOpen={menu === s.id} editing={editing === s.id}
                  onOpen={() => onOpen(s.id)}
                  onMenu={(open) => setMenu(open ? s.id : null)}
                  onStartEdit={() => { setEditing(s.id); setMenu(null); }}
                  onEdit={(title) => { setEditing(null); if (title !== null) onRename(s.id, title); }}
                  onPin={() => { setMenu(null); onPin(s.id, !s.pinned); }}
                  onDelete={() => { setMenu(null); if (confirm(`删除会话“${s.title || '无标题'}”？只删本浏览器里的记录。`)) onDelete(s.id); }} />
              ))}
            </ul>
          </section>
        ))}
      </div>
      <p className="side-foot">会话只存在本浏览器；接口无鉴权，只在本机打开。</p>
    </nav>
  );
}

interface ItemProps {
  session: Session; current: boolean; menuOpen: boolean; editing: boolean;
  onOpen: () => void; onMenu: (open: boolean) => void; onStartEdit: () => void;
  onEdit: (title: string | null) => void; onPin: () => void; onDelete: () => void;
}

function SessionItem({ session, current, menuOpen, editing, onOpen, onMenu, onStartEdit, onEdit, onPin, onDelete }: ItemProps) {
  const box = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (!menuOpen) return;
    const close = (e: Event) => { if (!box.current?.contains(e.target as Node)) onMenu(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onMenu(false); };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', esc); };
  }, [menuOpen, onMenu]);

  const title = session.title || '（无标题）';
  return (
    <li ref={box} className={`session${current ? ' current' : ''}${menuOpen ? ' menu-open' : ''}`}>
      {editing ? (
        <input className="rename" aria-label="会话标题" defaultValue={session.title} autoFocus maxLength={80}
          onBlur={(e) => onEdit(e.target.value.trim() || session.title)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            if (e.key === 'Escape') onEdit(null);
          }} />
      ) : (
        <button type="button" className="session-open" aria-current={current ? 'page' : undefined} onClick={onOpen}>
          {session.pinned && <span className="pin-mark" aria-label="已置顶"><IconPin /></span>}
          <span className="session-title">{title}</span>
        </button>
      )}
      {!editing && (
        <button type="button" className="icon-btn more" aria-label={`${title} 的操作`} aria-haspopup="menu"
          aria-expanded={menuOpen} onClick={() => onMenu(!menuOpen)}><IconMore /></button>
      )}
      {menuOpen && (
        <div className="menu" role="menu">
          <button type="button" role="menuitem" onClick={onStartEdit}><IconEdit /> 重命名</button>
          <button type="button" role="menuitem" onClick={onPin}><IconPin /> {session.pinned ? '取消置顶' : '置顶'}</button>
          <button type="button" role="menuitem" className="danger" onClick={onDelete}><IconTrash /> 删除</button>
        </div>
      )}
    </li>
  );
}
