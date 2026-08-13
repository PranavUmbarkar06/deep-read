/* ==========================================================================
   DEEP READ — FRONTEND JAVASCRIPT
   Connecting Chat Interface, Drag & Drop Uploads, & Agent API
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Element References
  const docList = document.getElementById('doc-list');
  const docEmpty = document.getElementById('doc-empty');
  const docCount = document.getElementById('doc-count');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const uploadProgress = document.getElementById('upload-progress');
  const progressBarFill = document.getElementById('progress-bar-fill');

  const docsPopover = document.getElementById('docs-popover');
  const btnDocsToggle = document.getElementById('btn-docs-toggle');
  const btnDocsClose = document.getElementById('btn-docs-close');

  const chatViewport = document.getElementById('chat-viewport');
  const chatMessages = document.getElementById('chat-messages');
  const welcomeHero = document.getElementById('welcome-hero');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const btnAttach = document.getElementById('btn-attach');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const activeIntentBadge = document.getElementById('active-intent-badge');
  const intentText = document.getElementById('intent-text');

  const attachedBar = document.getElementById('attached-bar');
  const attachedTags = document.getElementById('attached-tags');
  const btnClearAttached = document.getElementById('btn-clear-attached');

  // Modal References
  const docModal = document.getElementById('doc-modal');
  const modalClose = document.getElementById('modal-close');
  const modalDocTitle = document.getElementById('modal-doc-title');
  const modalDocName = document.getElementById('modal-doc-name');
  const modalDocSize = document.getElementById('modal-doc-size');
  const modalDocPages = document.getElementById('modal-doc-pages');
  const modalDocPath = document.getElementById('modal-doc-path');
  const modalBtnSummarize = document.getElementById('modal-btn-summarize');
  const modalBtnDelete = document.getElementById('modal-btn-delete');

  // State
  let uploadedDocuments = [];
  let selectedPdfPaths = new Set();
  let activeModalDoc = null;

  // Initialize
  fetchUploadedDocuments();
  setupAutoResizeTextarea();

  // =========================================================================
  // DOCUMENTS POPOVER
  // =========================================================================
  btnDocsToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    docsPopover.classList.toggle('hidden');
  });

  btnDocsClose.addEventListener('click', () => docsPopover.classList.add('hidden'));

  document.addEventListener('click', (e) => {
    if (!docsPopover.classList.contains('hidden') &&
        !docsPopover.contains(e.target) &&
        e.target !== btnDocsToggle && !btnDocsToggle.contains(e.target)) {
      docsPopover.classList.add('hidden');
    }
  });

  // =========================================================================
  // DOCUMENT MANAGEMENT & UPLOAD HANDLERS
  // =========================================================================
  async function fetchUploadedDocuments() {
    try {
      const res = await fetch('/api/documents');
      const data = await res.json();
      uploadedDocuments = data.documents || [];
      renderDocumentList();
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  }

  function renderDocumentList() {
    docCount.textContent = uploadedDocuments.length;

    if (uploadedDocuments.length === 0) {
      docEmpty.style.display = 'block';
      docList.querySelectorAll('.doc-item').forEach(el => el.remove());
      updateAttachedBar();
      return;
    }

    docEmpty.style.display = 'none';
    docList.querySelectorAll('.doc-item').forEach(el => el.remove());

    uploadedDocuments.forEach(doc => {
      const isChecked = selectedPdfPaths.has(doc.path);

      const item = document.createElement('div');
      item.className = `doc-item ${isChecked ? 'selected' : ''}`;
      item.innerHTML = `
        <div class="doc-item-left">
          <input type="checkbox" class="doc-checkbox" ${isChecked ? 'checked' : ''}>
          <div class="doc-info">
            <div class="doc-name" title="${doc.name}">${doc.name}</div>
            <div class="doc-meta">${doc.pages || '?'} pages &bull; ${formatBytes(doc.size)}</div>
          </div>
        </div>
        <div class="doc-actions">
          <button type="button" class="btn-doc-action btn-view-doc" title="View details"><i class="fa-solid fa-arrow-up-right-from-square"></i></button>
          <button type="button" class="btn-doc-action btn-del-doc" title="Delete file"><i class="fa-solid fa-trash"></i></button>
        </div>
      `;

      // Checkbox click
      const checkbox = item.querySelector('.doc-checkbox');
      checkbox.addEventListener('change', (e) => {
        e.stopPropagation();
        if (checkbox.checked) {
          selectedPdfPaths.add(doc.path);
          item.classList.add('selected');
        } else {
          selectedPdfPaths.delete(doc.path);
          item.classList.remove('selected');
        }
        updateAttachedBar();
      });

      // Click card to open modal
      item.addEventListener('click', (e) => {
        if (!e.target.closest('.doc-checkbox') && !e.target.closest('.btn-del-doc')) {
          openDocModal(doc);
        }
      });

      // Delete button
      const btnDel = item.querySelector('.btn-del-doc');
      btnDel.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteDocument(doc.name);
      });

      docList.appendChild(item);
    });

    updateAttachedBar();
  }

  function updateAttachedBar() {
    if (selectedPdfPaths.size === 0) {
      attachedBar.classList.add('hidden');
      return;
    }

    attachedBar.classList.remove('hidden');
    attachedTags.innerHTML = '';

    selectedPdfPaths.forEach(path => {
      const doc = uploadedDocuments.find(d => d.path === path);
      const name = doc ? doc.name : path.split(/[\/\\]/).pop();

      const tag = document.createElement('span');
      tag.className = 'pdf-tag';
      tag.textContent = name;
      attachedTags.appendChild(tag);
    });
  }

  btnClearAttached.addEventListener('click', () => {
    selectedPdfPaths.clear();
    renderDocumentList();
  });

  // Drag & Drop Listeners
  btnBrowse.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target !== btnBrowse) fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload(files);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileUpload(fileInput.files);
  });

  btnAttach.addEventListener('click', () => {
    docsPopover.classList.remove('hidden');
    fileInput.click();
  });

  async function handleFileUpload(fileList) {
    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      if (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')) {
        formData.append('files', f);
      }
    }

    uploadProgress.style.display = 'block';
    progressBarFill.style.width = '30%';

    try {
      progressBarFill.style.width = '70%';
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      progressBarFill.style.width = '100%';
      setTimeout(() => { uploadProgress.style.display = 'none'; progressBarFill.style.width = '0%'; }, 500);

      if (data.uploaded) {
        data.uploaded.forEach(u => selectedPdfPaths.add(u.path));
        await fetchUploadedDocuments();
      }
    } catch (err) {
      console.error('Upload failed:', err);
      uploadProgress.style.display = 'none';
      alert('Upload failed. Please try again.');
    }
  }

  async function deleteDocument(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (res.ok) {
        const doc = uploadedDocuments.find(d => d.name === filename);
        if (doc) selectedPdfPaths.delete(doc.path);
        await fetchUploadedDocuments();
      }
    } catch (err) {
      console.error('Failed to delete document:', err);
    }
  }

  // Modal Handlers
  function openDocModal(doc) {
    activeModalDoc = doc;
    modalDocTitle.textContent = doc.name;
    modalDocName.textContent = doc.name;
    modalDocSize.textContent = formatBytes(doc.size);
    modalDocPages.textContent = `${doc.pages || '?'} pages`;
    modalDocPath.textContent = doc.path;
    docModal.classList.remove('hidden');
  }

  modalClose.addEventListener('click', () => docModal.classList.add('hidden'));
  modalBtnDelete.addEventListener('click', () => {
    if (activeModalDoc) {
      deleteDocument(activeModalDoc.name);
      docModal.classList.add('hidden');
    }
  });

  modalBtnSummarize.addEventListener('click', () => {
    if (activeModalDoc) {
      selectedPdfPaths.add(activeModalDoc.path);
      updateAttachedBar();
      docModal.classList.add('hidden');
      docsPopover.classList.add('hidden');
      sendChatQuery(`Summarize the paper ${activeModalDoc.name}`);
    }
  });

  // =========================================================================
  // CHAT WORKSPACE & AGENT COMMUNICATOR
  // =========================================================================
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (query) {
      sendChatQuery(query);
      userInput.value = '';
      userInput.style.height = 'auto';
    }
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  btnClearChat.addEventListener('click', () => {
    chatMessages.innerHTML = '';
    welcomeHero.style.display = 'block';
    activeIntentBadge.classList.add('hidden');
  });

  // Sample prompt cards on the welcome hero
  window.useSampleQuery = function(text) {
    sendChatQuery(text);
  };

  async function sendChatQuery(query) {
    welcomeHero.style.display = 'none';

    // 1. Render User Message
    renderUserMessage(query);

    // 2. Render Loading Indicator
    const loadingRow = renderLoadingMessage();
    scrollToBottom();

    // 3. Prepare Payload
    const activePaths = Array.from(selectedPdfPaths);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          pdf_paths: activePaths
        })
      });

      const data = await res.json();
      loadingRow.remove();

      if (data.error) {
        renderAgentErrorMessage(data.error);
      } else {
        renderAgentResponse(data);
      }
    } catch (err) {
      console.error('Chat error:', err);
      loadingRow.remove();
      renderAgentErrorMessage('Failed to connect to Deep Read server. Make sure app.py is running.');
    }

    scrollToBottom();
  }

  // =========================================================================
  // MESSAGE RENDERERS
  // =========================================================================
  function renderUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    row.innerHTML = `
      <div class="message-content">
        <div class="user-bubble">${escapeHtml(text)}</div>
      </div>
      <div class="avatar user-avatar"><i class="fa-solid fa-user"></i></div>
    `;
    chatMessages.appendChild(row);
  }

  function renderLoadingMessage() {
    const row = document.createElement('div');
    row.className = 'message-row agent-row';
    row.innerHTML = `
      <div class="avatar agent-avatar"><i class="fa-solid fa-asterisk" style="font-size:0.6rem"></i></div>
      <div class="message-content">
        <div class="agent-card">
          <div class="loading-indicator">
            <span class="gen-cursor"></span>
            <span>Deep Read is reasoning through the graph&hellip;</span>
          </div>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    return row;
  }

  function renderAgentErrorMessage(errMsg) {
    const row = document.createElement('div');
    row.className = 'message-row agent-row';
    row.innerHTML = `
      <div class="avatar agent-avatar" style="color:var(--danger); border-color:var(--danger)"><i class="fa-solid fa-triangle-exclamation"></i></div>
      <div class="message-content">
        <div class="agent-card" style="border-color: var(--danger)">
          <div style="color:var(--danger); font-weight:700; font-size:0.85rem;">Pipeline error</div>
          <p style="margin-top:6px; font-size:0.85rem; color:var(--ink-soft);">${escapeHtml(errMsg)}</p>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
  }

  function renderAgentResponse(data) {
    // Update Header Intent Badge
    if (data.intent) {
      intentText.textContent = data.intent.toUpperCase();
      activeIntentBadge.classList.remove('hidden');
    }

    const row = document.createElement('div');
    row.className = 'message-row agent-row';

    let contentHtml = '';

    // 1. Intent Reason Banner
    if (data.router_reasoning) {
      contentHtml += `
        <div class="intent-reason-banner">
          <strong>${data.intent?.toUpperCase()}</strong> &nbsp;&bull;&nbsp; ${escapeHtml(data.router_reasoning)}
        </div>
      `;
    }

    // 2. Compatibility Alert (for compare intent)
    if (data.is_compatible !== undefined && data.is_compatible !== null) {
      const isComp = data.is_compatible;
      contentHtml += `
        <div class="compat-alert ${isComp ? 'compatible' : 'incompatible'}">
          <div>
            <strong>Gatecheck: ${isComp ? 'Compatible' : 'Incompatible'}</strong>
            <p>${escapeHtml(data.compatibility_reason || '')}</p>
          </div>
        </div>
      `;
    }

    // 3. Final Message / Markdown Content
    if (data.final_message) {
      let textToRender = data.final_message;

      // Try parsing JSON summary matrices if returned as JSON string
      if (typeof textToRender === 'string' && textToRender.trim().startsWith('{')) {
        try {
          const parsed = JSON.parse(textToRender);
          contentHtml += renderStructuredComparison(parsed);
          textToRender = null;
        } catch (e) {
          // Normal markdown
        }
      }

      if (textToRender) {
        contentHtml += `<div class="markdown-body">${marked.parse(textToRender)}</div>`;
      }
    }

    // 4. Paper Discovery Grid (for discover intent)
    if (data.papers && data.papers.length > 0) {
      contentHtml += `<h4 style="margin-top:16px; font-family:'Archivo',sans-serif; color:var(--ink); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px;">Discovered Papers (${data.papers.length})</h4>`;
      contentHtml += `<div class="paper-cards-grid">`;
      data.papers.forEach(p => {
        contentHtml += `
          <div class="paper-card">
            <div>
              <div class="paper-card-title">${escapeHtml(p.title)}</div>
              <div class="paper-card-summary">${escapeHtml(p.summary || 'No summary available.')}</div>
            </div>
            <div class="paper-card-footer">
              <a href="${p.url}" target="_blank" rel="noopener" class="btn-arxiv-link">arXiv PDF &rarr;</a>
            </div>
          </div>
        `;
      });
      contentHtml += `</div>`;
    }

    row.innerHTML = `
      <div class="avatar agent-avatar"><i class="fa-solid fa-asterisk" style="font-size:0.6rem"></i></div>
      <div class="message-content">
        <div class="agent-card">
          ${contentHtml}
        </div>
      </div>
    `;

    chatMessages.appendChild(row);
  }

  function renderStructuredComparison(obj) {
    let html = '';

    if (obj.matrix_summary) {
      html += `<div class="markdown-body"><p>${marked.parse(obj.matrix_summary)}</p></div>`;
    }

    if (obj.key_tradeoffs && Array.isArray(obj.key_tradeoffs)) {
      html += `
        <div class="matrix-container">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Dimension</th>
                <th>Paper / Approach 1</th>
                <th>Paper / Approach 2</th>
              </tr>
            </thead>
            <tbody>
      `;

      obj.key_tradeoffs.forEach(t => {
        html += `
          <tr>
            <td><strong>${escapeHtml(t.dimension || '')}</strong></td>
            <td>${escapeHtml(t.paper_1_takeaway || '')}</td>
            <td>${escapeHtml(t.paper_2_takeaway || '')}</td>
          </tr>
        `;
      });

      html += `</tbody></table></div>`;
    }

    if (obj.verdict) {
      html += `
        <div class="verdict-callout">
          <h4>Synthesis &amp; Recommendation</h4>
          <p style="font-size:0.85rem; color:var(--ink); margin:0;">${escapeHtml(obj.verdict)}</p>
        </div>
      `;
    }

    return html;
  }

  // =========================================================================
  // UTILITIES
  // =========================================================================
  function setupAutoResizeTextarea() {
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function formatBytes(bytes, decimals = 1) {
    if (!bytes) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});