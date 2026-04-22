(function () {
  const html = document.documentElement;
  const zhBtn = document.getElementById('lang-zh');
  const enBtn = document.getElementById('lang-en');
  const repoLink = document.getElementById('repo-link');
  const key = 'pg_retrieval_engine_site_lang';

  function setLang(lang) {
    html.lang = lang === 'en' ? 'en' : 'zh-CN';
    zhBtn.classList.toggle('active', html.lang === 'zh-CN');
    enBtn.classList.toggle('active', html.lang === 'en');
    localStorage.setItem(key, html.lang);
  }

  function detectInitialLang() {
    const queryLang = new URLSearchParams(window.location.search).get('lang');
    if (queryLang === 'en' || queryLang === 'zh') return queryLang === 'en' ? 'en' : 'zh-CN';

    const saved = localStorage.getItem(key);
    if (saved === 'en' || saved === 'zh-CN') return saved;

    const browser = navigator.language || '';
    return browser.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en';
  }

  zhBtn.addEventListener('click', function () {
    setLang('zh-CN');
  });

  enBtn.addEventListener('click', function () {
    setLang('en');
  });

  function detectRepoUrl() {
    const host = window.location.hostname || '';
    const pathParts = window.location.pathname.split('/').filter(Boolean);

    if (host.endsWith('.github.io')) {
      const owner = host.split('.')[0];
      const repo = pathParts.length > 0 ? pathParts[0] : owner + '.github.io';
      return 'https://github.com/' + owner + '/' + repo;
    }

    return 'https://github.com';
  }

  const repoUrl = detectRepoUrl();
  if (repoLink) {
    repoLink.href = repoUrl;
  }

  document.querySelectorAll('[data-gh-path]').forEach(function (el) {
    const filePath = el.getAttribute('data-gh-path');
    if (!filePath) return;
    el.href = repoUrl + '/blob/main/' + filePath;
  });

  setLang(detectInitialLang());
})();
