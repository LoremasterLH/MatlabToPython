Prevod prve polovice vaj pri predmetu ROSIS iz Matlab v Python 3.

<p>Uporabljen IDE:	PyCharm</p>
<p>Verzija Python:	3.6.1</p>
<p>Knjižnice:</p>
<ul>
	<li>matplotlib</li>
	<li>numpy</li>
	<li>pathlib</li>
	<li>Pillow (not PIL, as that is older unsupported version, Pillow is reffered to as PIL for the sake of imports)</li>
	<li>pylab</li>
	<li>scipy</li>
	<li>sounddevice</li>
	<li>time</li>
	<li>pytftb (from https://github.com/scikit-signal/pytftb)</li>
</ul>

<p>1-Vzorčenje				Prevedeno</p>
<p>2-Linearni Sistemi in Konvolucija	Prevedeno</p>
<p>3-Fourirjeva Analiza			Prevedeno</p>
<p>4-Lastnosti Fourirjeve Analize	Delno prevedeno</p>
<p>5-Filtri				Work in progress</p>

----------------------------------------------------------------------------------------------------

<h2>Manual Install</h2>
<ul>
	<li>Download and install latest version of Python 3.x</li>
	<li>Install PyCharm Community Edition</li>
	<li>Clone GitHub repository</li>
	<li>Open/Create a project in the root of the GitHub repository</li>
	<li>Go to File -> Settings -> Project: <name> -> Project interpreter</li>
		<li>Make sure Python 3.x is set as the project interpreter.</li>
		<li>Click on the plus on the right of the window and install all above mentioned packages (or any that are in imports of the files).</li>
	<li>Make sure to always execute (alt+shift+E) imports commands at the top before attempting to execute code.</li>
</ul>

<h2>Easy setup</h2>
Not thoroughly tested as I don't use Anaconda. Should be an easier setup than starting from scratch, though:

<h2>Install Anaconda</h2>

<p>If you want to use VS Code, install it once prompted (recommended if you're unfamiliar with Python, but hey, you can use Notepad, if you want)/p>
	
<h2>Install Finished</h2>

Open Anaconda Navigator (https://docs.anaconda.com/anaconda/user-guide/getting-started)</br>
Select VS Code</br>
File -> Open File -> Whatever file you want</br>
	
If you see horrible curly lines below imports do not fret! While Anaconda includes many packages, it does not include everything we need. Installing packages is easy however.</br>
To install packages that were not found:
Open "Anaconda Prompt"<br/>
	pip install <package>

In case you're wondering why sometimes pip and sometimes conda for installing: pip is Python's package manager while conda is package manager of Anaconda distribution.
Known issues:
Installing: If you get red text saying "distributed 1.21.8 requires msgpack, which is not installed."; don't worry. You don't need it. As long as you get "Successfully installed <package> you're good". You can however install it with "conda install -c anaconda msgpack-python".

To get it running:
Vsiew -> Command Paletter (Ctrl+Shift+P)

Windows:
In case you get strange errors after selecting "Run Selection/Line in Python Terminal" (shift+enter) such as:
= does not exist or is disconnected
Unable to initialize device PRN
It means Visual Studio Code starter using PowerShell for whatever reason.

To change PowerShell to cmd.exe:
In VS Code go to Settings (Ctrl+,)
Add the following to the right side (USER SETTINGS)
"terminal.integrated.shell.windows": "C:\\WINDOWS\\System32\\cmd.exe",
"terminal.integrated.shellArgs.windows": ["/K", "C:\\cmder\\vscode.bat"]

If you quit python from within the terminal, you have to rerun it (or restart VS Code).

PyCharm:
If there's no interpreter configured click on "Configure Python interpreter" and enable " Alternatively you can create a new project from the file and Anaconda should be picked by default.

If you get errors such as "numpy" not found, make sure your interpreter is set to anaconda3 python:
C:\Users\martink\AppData\Local\Continuum\anaconda3\python.exe (changing interpreter takes time)

For each grayed out "import" and "from" click on it 

Not all needed packages are included in Anaconda. To install missing packages (ones which have import/for statements in grey) go to:
File->Settings (Ctrl+Alt+S) click on + on the right side (Ctrl+Insert)
Potentially needed imports for 1st assignment:
<ul>
	<li>pip (to update to recent)</li>
	<li>plotly</li>
	<li>Pillow (not PIL)</li>
</ul>
