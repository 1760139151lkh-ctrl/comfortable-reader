/* Use the same character-location algorithm as the installed EPUB reader. */
const fs=require('node:fs'),path=require('node:path');
const {DOMParser}=require('../desktop/node_modules/@xmldom/xmldom');
const Locations=require('../desktop/node_modules/epubjs/lib/locations').default;
const input=JSON.parse(fs.readFileSync(0,'utf8')),locations=[];
const parser={createRange:Locations.prototype.createRange,break:1000};
for(const [ordinal,item] of input.spine.entries()){
 if(item.linear===false)continue;
 const doc=new DOMParser().parseFromString(fs.readFileSync(path.join(input.root,item.path),'utf8'),'application/xhtml+xml');
 global.document=doc;global.NodeFilter={SHOW_TEXT:4};
 locations.push(...Locations.prototype.parse.call(parser,doc.documentElement,`/${((input.spineNodeIndex??2)+1)*2}/${(ordinal+1)*2}`,1000));
}
process.stdout.write(JSON.stringify({version:'epubjs-locations-1000@1',locations}));
