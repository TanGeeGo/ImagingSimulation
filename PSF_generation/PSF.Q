[General]
SyntaxVersion=2
BeginHotkey=121
BeginHotkeyMod=0
PauseHotkey=0
PauseHotkeyMod=0
StopHotkey=123
StopHotkeyMod=0
RunOnce=1
EnableWindow=
MacroID=92c2045a-680f-49a7-8c60-82e778c23784
Description=PSF
Enable=1
AutoRun=0
[Repeat]
Type=0
Number=1
[SetupUI]
Type=2
QUI=
[Relative]
SetupOCXFile=
[Comment]

[Script]
//512波段//
Dim low_512_wavindex
low_512_wavindex = 40
Dim upp_512_wavindex
upp_512_wavindex = 42
//256波段//
Dim low_256_wavindex
low_256_wavindex = 43
Dim upp_256_wavindex
upp_256_wavindex = 46
//128_1波段//
Dim low_128_wavindex_fir
low_128_wavindex_fir = 47
Dim upp_128_wavindex_fir
upp_128_wavindex_fir = 49
//64_1波段//
Dim low_64_wavindex_fir
low_64_wavindex_fir = 50
Dim upp_64_wavindex_fir
upp_64_wavindex_fir = 59
//128_3波段//
Dim low_128_wavindex_trd
low_128_wavindex_trd = 60
Dim upp_128_wavindex_trd
upp_128_wavindex_trd = 63
//64_2波段//
Dim low_64_wavindex_scd
low_64_wavindex_scd = 0
Dim upp_64_wavindex_scd
upp_64_wavindex_scd = 0
//128_2波段//
Dim low_128_wavindex_scd
low_128_wavindex_scd = 64
Dim upp_128_wavindex_scd
upp_128_wavindex_scd = 70


//512运算延时//
Dim time_delay_512
time_delay_512 = 130000
//256运算延时//
Dim time_delay_256
time_delay_256 = 40000
//128运算延时//
Dim time_delay_128
time_delay_128 = 15000
//64运算延时//
Dim time_delay_64
time_delay_64 = 5000

Dim fld_00_value
fld_00_value = "0.15"
Dim fld_01_value
fld_01_value = "7.05"
Dim fld_02_value
fld_02_value = "7.10"
Dim fld_03_value
fld_03_value = "7.15"
Dim fld_04_value
fld_04_value = "7.20"
Dim fld_05_value
fld_05_value = "7.25"
Dim fld_06_value
fld_06_value = "7.30"
Dim fld_07_value
fld_07_value = "7.35"
Dim fld_08_value
fld_08_value = "7.40"
Dim fld_09_value
fld_09_value = "7.45"
Dim fld_10_value
fld_10_value = "7.50"
Dim fld_11_value
fld_11_value = "7.55"
Dim fld_12_value
fld_12_value = "7.60"
Dim fld_13_value
fld_13_value = "7.65"
Dim fld_14_value
fld_14_value = "7.70"
Dim fld_15_value
fld_15_value = "7.75"
Dim fld_16_value
fld_16_value = "7.80"
Dim fld_17_value
fld_17_value = "7.85"
Dim fld_18_value
fld_18_value = "7.90"
Dim fld_19_value
fld_19_value = "7.95"
Dim fld_20_value
fld_20_value = "8.00"
Dim fld_21_value
fld_21_value = "8.05"
Dim fld_22_value
fld_22_value = "8.10"
Dim fld_23_value
fld_23_value = "8.15"
Dim fld_24_value
fld_24_value = "8.20"
Dim fld_25_value
fld_25_value = "8.25"
Dim fld_26_value
fld_26_value = "8.30"
Dim fld_27_value
fld_27_value = "8.35"
Dim fld_28_value
fld_28_value = "8.40"
Dim fld_29_value
fld_29_value = "8.45"

Dim fld_00_folder
fld_00_folder = "PSF_info_fld_00_15"
Dim fld_01_folder
fld_01_folder = "PSF_info_fld_07_05"
Dim fld_02_folder
fld_02_folder = "PSF_info_fld_07_10"
Dim fld_03_folder
fld_03_folder = "PSF_info_fld_07_15"
Dim fld_04_folder
fld_04_folder = "PSF_info_fld_07_20"
Dim fld_05_folder
fld_05_folder = "PSF_info_fld_07_25"
Dim fld_06_folder
fld_06_folder = "PSF_info_fld_07_30"
Dim fld_07_folder
fld_07_folder = "PSF_info_fld_07_35"
Dim fld_08_folder
fld_08_folder = "PSF_info_fld_07_40"
Dim fld_09_folder
fld_09_folder = "PSF_info_fld_07_45"
Dim fld_10_folder
fld_10_folder = "PSF_info_fld_07_50"
Dim fld_11_folder
fld_11_folder = "PSF_info_fld_07_55"
Dim fld_12_folder
fld_12_folder = "PSF_info_fld_07_60"
Dim fld_13_folder
fld_13_folder = "PSF_info_fld_07_65"
Dim fld_14_folder
fld_14_folder = "PSF_info_fld_07_70"
Dim fld_15_folder
fld_15_folder = "PSF_info_fld_07_75"
Dim fld_16_folder
fld_16_folder = "PSF_info_fld_07_80"
Dim fld_17_folder
fld_17_folder = "PSF_info_fld_07_85"
Dim fld_18_folder
fld_18_folder = "PSF_info_fld_07_90"
Dim fld_19_folder
fld_19_folder = "PSF_info_fld_07_95"
Dim fld_20_folder
fld_20_folder = "PSF_info_fld_08_00"
Dim fld_21_folder
fld_21_folder = "PSF_info_fld_08_05"
Dim fld_22_folder
fld_22_folder = "PSF_info_fld_08_10"
Dim fld_23_folder
fld_23_folder = "PSF_info_fld_08_15"
Dim fld_24_folder
fld_24_folder = "PSF_info_fld_08_20"
Dim fld_25_folder
fld_25_folder = "PSF_info_fld_08_25"
Dim fld_26_folder
fld_26_folder = "PSF_info_fld_08_30"
Dim fld_27_folder
fld_27_folder = "PSF_info_fld_08_35"
Dim fld_28_folder
fld_28_folder = "PSF_info_fld_08_40"
Dim fld_29_folder
fld_29_folder = "PSF_info_fld_08_45"

//视场索引
Dim fld_index
fld_index = 0

/////////////////////////////////////////////////////////////////////////////////////
Dim wav_y_value
wav_y_value = 670
//十个视场//
For 10
	//改动视场//
	MoveTo 545, 55
	LeftClick 1
	Delay 500
	MoveTo 1118, 794
	LeftDoubleClick 1
	Delay 500
	If fld_index = 0 Then 
		SayString fld_00_value
	ElseIf fld_index = 1 Then 
		SayString fld_01_value
	ElseIf fld_index = 2 Then 
		SayString fld_02_value
	ElseIf fld_index = 3 Then 
		SayString fld_03_value
	ElseIf fld_index = 4 Then 
		SayString fld_04_value
	ElseIf fld_index = 5 Then 
		SayString fld_05_value
	ElseIf fld_index = 6 Then 
		SayString fld_06_value
	ElseIf fld_index = 7 Then 
		SayString fld_07_value
	ElseIf fld_index = 8 Then 
		SayString fld_08_value
	ElseIf fld_index = 9 Then 
		SayString fld_09_value
	ElseIf fld_index = 10 Then 
		SayString fld_10_value
	ElseIf fld_index = 11 Then 
		SayString fld_11_value
	ElseIf fld_index = 12 Then 
		SayString fld_12_value
	ElseIf fld_index = 13 Then 
		SayString fld_13_value
	ElseIf fld_index = 14 Then 
		SayString fld_14_value
	ElseIf fld_index = 15 Then 
		SayString fld_15_value
	ElseIf fld_index = 16 Then 
		SayString fld_16_value
	ElseIf fld_index = 17 Then 
		SayString fld_17_value
	ElseIf fld_index = 18 Then 
		SayString fld_18_value
	ElseIf fld_index = 19 Then 
		SayString fld_19_value
	ElseIf fld_index = 20 Then 
		SayString fld_20_value
	ElseIf fld_index = 21 Then 
		SayString fld_21_value
	ElseIf fld_index = 22 Then 
		SayString fld_22_value
	ElseIf fld_index = 23 Then 
		SayString fld_23_value
	ElseIf fld_index = 24 Then 
		SayString fld_24_value
	ElseIf fld_index = 25 Then 
		SayString fld_25_value
	ElseIf fld_index = 26 Then 
		SayString fld_26_value
	ElseIf fld_index = 27 Then 
		SayString fld_27_value
	ElseIf fld_index = 28 Then 
		SayString fld_28_value
	ElseIf fld_index = 29 Then 
		SayString fld_29_value
	End If
	Delay 500
	KeyPress "Enter", 1
	Delay 500
    //确认波长信息//
    MoveTo 580, 60
    LeftClick 1
    Delay 500
    //570波长//
    MoveTo 1360, wav_y_value
    LeftDoubleClick 1
    SayString "0.57"
    Delay 500
    //580波长//
    MoveTo 1360, wav_y_value+25
    LeftDoubleClick 1
    SayString "0.58"
    Delay 500
    //590波长//
    MoveTo 1360, wav_y_value+25*2
    LeftDoubleClick 1
    SayString "0.59"
    Delay 500
    //600波长//
    MoveTo 1360, wav_y_value+25*3
    LeftDoubleClick 1
    SayString "0.60"
    Delay 500
    //610波长//
    MoveTo 1360, wav_y_value+25*4
    LeftDoubleClick 1
    SayString "0.61"
    Delay 500
    //620波长//
    MoveTo 1360, wav_y_value+25*5
    LeftDoubleClick 1
    SayString "0.62"
    Delay 500
    //630波长//
    MoveTo 1360, wav_y_value+25*6
    LeftDoubleClick 1
    SayString "0.63"
    Delay 500
    //ok//
    KeyPress "Enter", 1
    Delay 500


    /////////////////////////////////////////////////////////////////////////////////////
    //开始计算波长PSF//
    Dim wav_index
    wav_index = 40
    For 24
        //调出setting面板//
        MoveTo 300, 1050
        Delay 500
        RightClick 1
        Delay 500
        //调整波长//
        MoveTo 1547, 648
        Delay 500
        LeftClick 1
        Delay 500
        KeyPress "Down", 1
        Delay 500
        KeyPress "Enter", 1
        Delay 500
        //由波长索引判断采样范围//
        MoveTo 1260, 675
        Delay 500
        LeftClick 1
        Delay 500
        If (low_512_wavindex <= wav_index)and(wav_index <= upp_512_wavindex) Then 
            //采样为512*512//
            MoveTo 1200, 745
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//

            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_512
        ElseIf (low_256_wavindex <= wav_index) and (wav_index <= upp_256_wavindex) Then
            //采样256*256//
            MoveTo 1200, 730
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_256
        ElseIf (low_128_wavindex_fir <= wav_index) and (wav_index <= upp_128_wavindex_fir) Then
            //采样128*128//
            MoveTo 1200, 715
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_128
        ElseIf (low_64_wavindex_fir <= wav_index) and (wav_index <= upp_64_wavindex_fir) Then
            //采样64*64//
            MoveTo 1200, 700
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_64
        ElseIf (low_128_wavindex_trd <= wav_index) and (wav_index <= upp_128_wavindex_trd) Then
            //采样64*64//
            MoveTo 1200, 715
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_128
        End If
        //调出Windows，保存//
        MoveTo 230, 870
        Delay 500
        LeftClick 1
        Delay 500
        MoveTo 270, 910
        Delay 500
        LeftClick 1
        Delay 500
        //txt名字//
        If wav_index = 40 Then
            SayString "wav_040"
        ElseIf wav_index = 41 Then
            SayString "wav_041"
        ElseIf wav_index = 42 Then
            SayString "wav_042"
        ElseIf wav_index = 43 Then
            SayString "wav_043"
        ElseIf wav_index = 44 Then
            SayString "wav_044"
        ElseIf wav_index = 45 Then
            SayString "wav_045"
        ElseIf wav_index = 46 Then
            SayString "wav_046"
        ElseIf wav_index = 47 Then
            SayString "wav_047"
        ElseIf wav_index = 48 Then
            SayString "wav_048"
        ElseIf wav_index = 49 Then
            SayString "wav_049"
        ElseIf wav_index = 50 Then
            SayString "wav_050"
        ElseIf wav_index = 51 Then
            SayString "wav_051"
        ElseIf wav_index = 52 Then
            SayString "wav_052"
        ElseIf wav_index = 53 Then
            SayString "wav_053"
        ElseIf wav_index = 54 Then
            SayString "wav_054"
        ElseIf wav_index = 55 Then
            SayString "wav_055"
        ElseIf wav_index = 56 Then
            SayString "wav_056"
        ElseIf wav_index = 57 Then
            SayString "wav_057"
        ElseIf wav_index = 58 Then
            SayString "wav_058"
        ElseIf wav_index = 59 Then
            SayString "wav_059"
        ElseIf wav_index = 60 Then
            SayString "wav_060"
        ElseIf wav_index = 61 Then
            SayString "wav_061"
        ElseIf wav_index = 62 Then
            SayString "wav_062"
        ElseIf wav_index = 63 Then
            SayString "wav_063"
        End If
        //敲回车保存//
        KeyPress "Enter", 1
        Delay 500
        //敲回车确认//

        KeyPress "Enter", 1
        Delay 500
        //波长索引+1//
        wav_index = wav_index + 1
        Delay 500
    Next

    /////////////////////////////////////////////////////////////////////////////////////
    //改波长调出波长界面确认波长信息//
    MoveTo 580, 60
    Delay 500
    LeftClick 1
    Delay 500
    //640波长//
    MoveTo 1360, wav_y_value
    Delay 500
    LeftDoubleClick 1
    SayString "0.64"
    //650波长//
    MoveTo 1360, wav_y_value+25
    Delay 500
    LeftDoubleClick 1
    SayString "0.65"
    //660波长//
    MoveTo 1360, wav_y_value+25*2
    Delay 500
    LeftDoubleClick 1
    SayString "0.66"
    //670波长//
    MoveTo 1360, wav_y_value+25*3
    Delay 500
    LeftDoubleClick 1
    SayString "0.67"
    //680波长//
    MoveTo 1360, wav_y_value+25*4
    Delay 500
    LeftDoubleClick 1
    SayString "0.68"
    //690波长//
    MoveTo 1360, wav_y_value+25*5
    Delay 500
    LeftDoubleClick 1
    SayString "0.69"
    //700波长//
    MoveTo 1360, wav_y_value+25*6
    Delay 500
    LeftDoubleClick 1
    SayString "0.70"
    //ok//
    KeyPress "Enter", 1
    Delay 500

    /////////////////////////////////////////////////////////////////////////////////////
    //开始计算波长PSF
    For 7
        //调出setting面板//
        MoveTo 300, 1050
        Delay 500
        RightClick 1
        Delay 500
        //调整波长//
        MoveTo 1547, 648
        Delay 500
        LeftClick 1
        Delay 500
        If wav_index = 64 Then 
            KeyPress "Up", 6
            Delay 500
        Else 
            KeyPress "Down", 1
            Delay 500
        End If
        KeyPress "Enter", 1
        Delay 500
        //由波长索引判断采样范围//
        MoveTo 1260, 675
        Delay 500
        LeftClick 1
        Delay 500
        If (low_64_wavindex_scd <= wav_index)and(wav_index <= upp_64_wavindex_scd) Then 
            //采样为64*64//
            MoveTo 1200, 700
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_64
        ElseIf (low_128_wavindex_scd <= wav_index)and(wav_index <= upp_128_wavindex_scd) Then 
            //采样为128*128//
            MoveTo 1200, 715
            Delay 500
            LeftClick 1
            Delay 500
            //确认当前setting//
            KeyPress "Enter", 1
            //延时，等待计算//
            Delay time_delay_128
        End If
        //调出Windows，保存//
        MoveTo 230, 870
        Delay 500
        LeftClick 1
        Delay 500
        MoveTo 270, 910
        Delay 500
        LeftClick 1
        Delay 500
        //txt名字//
        If wav_index = 64 Then
            SayString "wav_064"
        ElseIf wav_index = 65 Then
            SayString "wav_065"
        ElseIf wav_index = 66 Then
            SayString "wav_066"
        ElseIf wav_index = 67 Then
            SayString "wav_067"
        ElseIf wav_index = 68 Then
            SayString "wav_068"
        ElseIf wav_index = 69 Then
            SayString "wav_069"
        ElseIf wav_index = 70 Then
            SayString "wav_070"
        End If
        //敲回车保存//
        KeyPress "Enter", 1
        Delay 500
        //敲回车确认//
        KeyPress "Enter", 1
        Delay 500
        //波长索引+1//
        wav_index = wav_index + 1
        Delay 500
    Next

    /////////////////////////////////////////////////////////////////////////////////////////
    //将PSF整合进Excel中
    //调出文件资源管理器
    MoveTo 515, 1420
    LeftClick 1
    Delay 500
    MoveTo 500, 1200
    //创建Excel//
    RightClick 1
    Delay 500
    KeyPress "w", 1
    Delay 500
    MoveTo 850, 1395
    LeftClick 1
    Delay 500
    //重命名//
    SayString "PSF_info"
    Delay 500
    KeyPress "Enter", 1
    Delay 500
    //打开Excel//
    KeyPress "Enter", 1
    Delay 3000
    //导入数据索引//
    Dim input_index
    input_index = 1
    For 31
        // 找到文件夹
        If input_index = 1 Then 
            //点“数据”//
            MoveTo 320, 45
            LeftClick 1
            Delay 500
            LeftClick 1
            Delay 500
            //点“自文本”//
            MoveTo 122, 95
            LeftClick 1
            Delay 500
            //点“E盘”//
            MoveTo 76, 725
            LeftClick 1
            Delay 500
            //点“像差修复课题”//
            MoveTo 245, 807
            LeftDoubleClick 1
            Delay 500
            //鼠标滚轮滚到最下面
            MouseWheel - 5 
            Delay 500
            MouseWheel - 5 
            Delay 500
            MouseWheel - 5 
            Delay 500
            MouseWheel - 5 
            Delay 500
            MouseWheel - 5
            //点20200415
            MoveTo 233, 758
            LeftDoubleClick 1
            Delay 500
            //点第一个文件夹
            MoveTo 210, 140
            LeftDoubleClick 1
            Delay 500
            //点第一个波长
            MoveTo 244, 143
            LeftClick 1
            Delay 500
        Else 
            //点“数据”//
            MoveTo 320, 45
            LeftClick 1
            Delay 500
            //点“自文本”//
            MoveTo 122, 95
            LeftClick 1
            Delay 500
            //点第一个波长
            MoveTo 244, 143
            LeftClick 1
            Delay 500
            //敲index-1次down//
            KeyPress "Down", input_index - 1
        End If
        //敲回车//
        KeyPress "Enter", 1
        Delay 500

        KeyPress "Enter", 1
        Delay 500
        KeyPress "Enter", 1
        Delay 500
        KeyPress "Enter", 1
        Delay 500
        KeyPress "Enter", 1
        Delay 500
        //增加一个sheet
        If input_index < 31 Then 
            KeyDown 16, 1
            KeyPress 122, 1
            KeyUp 16, 1
            Delay 500
            input_index = input_index + 1
            Delay 500	
        End If
    Next
    //点击保存//
    MoveTo 25, 15
    LeftClick 1
    Delay 5000
    //关闭Excel//
    MoveTo 2535, 16
    LeftClick 1
    Delay 2000
    //选中所有文件//
    MoveTo 510, 600
    LeftClick 1
    Delay 500
    KeyDown 17, 1
	KeyPress 65, 1
	KeyUp 17, 1
	Delay 500
	//剪切//
	RightClick 1
	Delay 500
	MoveTo 585, 960
	LeftClick 1
	Delay 500
	//后退//
	MoveTo 94, 66
	LeftClick 1
	Delay 500
	//新建文件夹//
	MoveTo 1719, 1058
	RightClick 1
	Delay 500
	KeyPress "W", 1
	Delay 500
	KeyPress "F", 1
	Delay 500
	//输入名称//
	If fld_index = 0 Then 
		SayString fld_00_folder
	ElseIf fld_index = 1 Then
		SayString fld_01_folder
	ElseIf fld_index = 2 Then
		SayString fld_02_folder
	ElseIf fld_index = 3 Then
		SayString fld_03_folder
	ElseIf fld_index = 4 Then
		SayString fld_04_folder
	ElseIf fld_index = 5 Then
		SayString fld_05_folder
	ElseIf fld_index = 6 Then
		SayString fld_06_folder
	ElseIf fld_index = 7 Then
		SayString fld_07_folder
	ElseIf fld_index = 8 Then
		SayString fld_08_folder
	ElseIf fld_index = 9 Then
		SayString fld_09_folder
	ElseIf fld_index = 10 Then
		SayString fld_10_folder
	ElseIf fld_index = 11 Then
		SayString fld_11_folder
	ElseIf fld_index = 12 Then
		SayString fld_12_folder
	ElseIf fld_index = 13 Then
		SayString fld_13_folder
	ElseIf fld_index = 14 Then
		SayString fld_14_folder
	ElseIf fld_index = 15 Then
		SayString fld_15_folder
	ElseIf fld_index = 16 Then
		SayString fld_16_folder
	ElseIf fld_index = 17 Then
		SayString fld_17_folder
	ElseIf fld_index = 18 Then
		SayString fld_18_folder
	ElseIf fld_index = 19 Then
		SayString fld_19_folder
	ElseIf fld_index = 20 Then
		SayString fld_20_folder
	ElseIf fld_index = 21 Then
		SayString fld_21_folder
	ElseIf fld_index = 22 Then
		SayString fld_22_folder
	ElseIf fld_index = 23 Then
		SayString fld_23_folder
	ElseIf fld_index = 24 Then
		SayString fld_24_folder
	ElseIf fld_index = 25 Then
		SayString fld_25_folder
	ElseIf fld_index = 26 Then
		SayString fld_26_folder
	ElseIf fld_index = 27 Then
		SayString fld_27_folder
	ElseIf fld_index = 28 Then
		SayString fld_28_folder
	ElseIf fld_index = 29 Then
		SayString fld_29_folder
	End If
    Delay 500
    //新建文件夹//
    KeyPress "Enter", 1
	Delay 500
	KeyPress "Enter", 1
	MoveTo 1089, 672
	LeftClick 1
	Delay 500
	//黏贴//
	KeyDown 17, 1
	KeyPress 86, 1
	KeyUp 17, 1
	//往上退，回到1文件夹//
	MoveTo 93, 66
	LeftClick 1
	Delay 500
	MoveTo 256, 126
	LeftDoubleClick 1
	Delay 500
	
	//调出zemax//
    MoveTo 710, 1420
    LeftClick 1 
	
	//调出setting面板
    MoveTo 300, 1050
    Delay 500
    RightClick 1
    Delay 500
    //调整波长to all//
    MoveTo 1547, 648
    Delay 500
    LeftClick 1
    Delay 500
    KeyPress "Up", 24
    Delay 500
    KeyPress "Enter", 1
    Delay 500
    //调整采样为32*32//
    MoveTo 1260, 675
    Delay 500
    LeftClick 1
    Delay 500
    MoveTo 1200, 685
    Delay 500
    LeftClick 1
    Delay 500
    KeyPress "Enter", 1
    Delay 30000
	// 视场加1//
	fld_index = fld_index + 1
Next