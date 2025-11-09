# Как пересобрать документацию

## Быстрая инструкция

```powershell
cd c:\Users\pasha\thefittest\docs
.\rebuild-docs.bat
cd ..
git add .
git commit -m "docs: update documentation"
git push
```

## Что делает rebuild-docs.bat

1. Очищает старую сборку (`make.bat clean`)
2. Собирает новую документацию (`make.bat html`)
3. Копирует HTML из `build/html/` в корень `docs/`
4. Удаляет временную папку `build/`

## После push

Подождите 1-2 минуты, затем откройте https://sherstpasha.github.io/thefittest/ в режиме инкогнито (Ctrl+Shift+N) или очистите кэш (Ctrl+F5).
