"""Specifically approved SIDD public validation archive acquisition; no training."""
from pathlib import Path
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import hashlib,json,sys,time,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parents[2]
EXPECTED=461780640


class Form(HTMLParser):
    """Read only the public large-file confirmation form, never account credentials."""
    def __init__(self):super().__init__();self.action=None;self.fields={};self.active=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a.get('action');self.active=True
        if tag=='input' and self.active and a.get('type')=='hidden' and a.get('name'):self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def main() -> None:
    """Download once from authors' public mirror, with exact size cap and CRC checks."""
    if sys.argv[1:]!=['--approved']:raise PermissionError('Specific SIDD acquisition approval required')
    directory=ROOT/'data/raw/SIDD_validation_archive';directory.mkdir(exist_ok=False)
    target=directory/'test.zip';partial=directory/'test.zip.partial'
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    url='https://drive.usercontent.google.com/download?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ&export=download&confirm=t'
    response=opener.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45)
    if 'text/html' in response.headers.get('Content-Type',''):
        text=response.read(256*1024).decode(errors='replace');response.close()
        form=Form();form.feed(text)
        if not form.action or urllib.parse.urlparse(form.action).hostname!='drive.usercontent.google.com':
            (directory/'download_diagnostic.html').write_text(text)
            raise RuntimeError('Public mirror requires unavailable confirmation/access; no quota or account workaround attempted')
        url=form.action+'?'+urllib.parse.urlencode(form.fields)
        response=opener.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45)
    if 'text/html' in response.headers.get('Content-Type',''):raise RuntimeError('Mirror returned HTML, not SIDD archive')
    start=time.monotonic();count=0;h=hashlib.sha256();next_progress=25*1024**2
    with response,partial.open('xb') as stream:
        for chunk in iter(lambda:response.read(1024*1024),b''):
            count+=len(chunk)
            if count>EXPECTED:raise ValueError('Download exceeded specifically approved byte limit; partial retained')
            if time.monotonic()-start>1800:raise TimeoutError('SIDD acquisition cap 30 minutes reached; partial retained')
            stream.write(chunk);h.update(chunk)
            if count>=next_progress:print(f'SIDD archive: {count/1024**2:.1f} MiB / {EXPECTED/1024**2:.1f}',flush=True);next_progress+=25*1024**2
    if count!=EXPECTED:raise ValueError('SIDD archive truncated; partial retained')
    partial.rename(target)
    sys.path.insert(0,str(ROOT/'notebooks'))
    from cloud_workflow import safe_members
    import zipfile
    with zipfile.ZipFile(target) as archive:
        safe_members(archive)
        if archive.testzip() is not None:raise ValueError('SIDD archive CRC failed')
        names=archive.namelist()
    receipt={'status':'acquired_crc_verified_not_imported','archive':str(target),'bytes':count,'sha256':h.hexdigest(),'seconds':time.monotonic()-start,'origin':'Authors Restormer public SIDD validation mirror','members':names,'spending_usd':0,'training':False}
    (ROOT/'reports/engineering_v2/sidd_acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('SIDD acquired and CRC verified',flush=True)


if __name__=='__main__':main()
