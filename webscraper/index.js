const rp = require('request-promise');
const cheerio = require('cheerio')
const fs = require('fs');
const path = require('path');

async function main() {
    console.log(process.argv);
    let term = process.argv[2];
    if (!term) {
        console.log('need search term arg');
    }
    let page = 1;
    let counter = 0;
    let songsToGet = process.argv[3] ? process.argv[3] : 200;
    const songFolder = path.join(__dirname, 'songs');
    if (!fs.existsSync(songFolder)) {
        fs.mkdirSync(songFolder, { recursive: true });
    }
    while (counter < songsToGet) {
        counter += await consumePage(page, term);
        page += 1;
    }
    return 'done';
}

async function consumePage(page, term) {
    const url = `http://www.midiworld.com/search/${page}/?q=${term}`;
    const data = await rp(url);
    const $ = cheerio.load(data);
    const lis = $('#page li').toArray();
    const songs = lis.map((li) => {
        if (li.children.length === 7) {
            return {
                name: li.children[0].data.trim(),
                url: li.children[1].attribs.href,
            };
        }
    }).filter(x => x);
    console.log(songs);
    for (let i = 0; i < songs.length; i++) {
        rp(songs[i].url).pipe(fs.createWriteStream(`${path.join(__dirname, 'songs')}/${songs[i].name}.mid`));
    }
    return songs.length;
}

main().then((result) => {
    console.log(result);
}).catch((err) => {
    console.error(err);
});