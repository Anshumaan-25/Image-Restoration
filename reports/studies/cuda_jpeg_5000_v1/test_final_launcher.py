"""Offline safeguards for the final evaluation wrapper; no protected inference."""
from pathlib import Path
import importlib.util,io,stat,subprocess,sys,unittest,zipfile
path=Path(__file__).with_name('colab_final_evaluation.py')
spec=importlib.util.spec_from_file_location('final_launcher',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FinalLauncherTests(unittest.TestCase):
    def archive(self,names):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            for name in names:z.writestr(name,b'fixture')
        stream.seek(0);return zipfile.ZipFile(stream)
    def test_valid_archive(self):
        with self.archive(['data/image.png','records.json']) as z:
            self.assertEqual(len(module.safe(z)),2)
    def test_unsafe_paths(self):
        for name in ['../outside','/absolute','data/../../outside','data\\outside']:
            with self.subTest(name=name),self.archive([name]) as z:
                with self.assertRaises(ValueError):module.safe(z)
    def test_duplicate_and_symlink(self):
        with self.archive(['same','same']) as z:
            with self.assertRaises(ValueError):module.safe(z)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(entry,'target')
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            with self.assertRaises(ValueError):module.safe(z)
    def test_approval_precedes_cloud_access(self):
        result=subprocess.run([sys.executable,str(path)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Specific protected-test and evaluation budget approval',result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
