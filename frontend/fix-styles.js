const fs = require('fs');
const path = require('path');

const dir = 'C:\\Scandium labs\\src\\components';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.tsx'));

for (const file of files) {
  const filePath = path.join(dir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  let originalContent = content;

  // Add inline styles for background and color
  content = content.replace(
    /<section([^>]*?)className="([^"]*?bg-background text-primary-text[^"]*?)"(.*?)style={{([^}]*)}}([^>]*?)>/g,
    '<section$1className="$2"$3style={{$4, background: \'var(--background)\', color: \'var(--primary-text)\' }}$5>'
  );

  content = content.replace(
    /<section([^>]*?)className="([^"]*?bg-background text-primary-text[^"]*?)"((?!style=)[^>]*?)>/g,
    '<section$1className="$2"$3 style={{ background: \'var(--background)\', color: \'var(--primary-text)\' }}>'
  );

  content = content.replace(
    /<footer([^>]*?)className="([^"]*?bg-background text-primary-text[^"]*?)"(.*?)style={{([^}]*)}}([^>]*?)>/g,
    '<footer$1className="$2"$3style={{$4, background: \'var(--background)\', color: \'var(--primary-text)\' }}$5>'
  );

  if (content !== originalContent) {
    fs.writeFileSync(filePath, content);
    console.log(`Updated ${file}`);
  }
}

// Update layout.tsx
const layoutPath = 'C:\\Scandium labs\\src\\app\\layout.tsx';
let layoutContent = fs.readFileSync(layoutPath, 'utf8');
if (!layoutContent.includes("style={{ background: 'var(--background)', color: 'var(--primary-text)' }}")) {
  layoutContent = layoutContent.replace(
    /<body className="([^"]*?bg-background text-primary-text[^"]*?)"([^>]*?)>/g,
    '<body className="$1"$2 style={{ background: \'var(--background)\', color: \'var(--primary-text)\' }}>'
  );
  fs.writeFileSync(layoutPath, layoutContent);
  console.log('Updated layout.tsx');
}
