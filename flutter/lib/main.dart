import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Video Stream Example',
      home: VideoScreen(),
    );
  }
}

class VideoScreen extends StatefulWidget {
  const VideoScreen({super.key});

  @override
  _VideoScreenState createState() => _VideoScreenState();
}

class _VideoScreenState extends State<VideoScreen> {
  VideoPlayerController? _controller;
  bool _isDownloading = false;
  final TextEditingController _urlController = TextEditingController();

  @override
  void dispose() {
    _controller?.dispose();
    _urlController.dispose();
    super.dispose();
  }

  Future<void> downloadAndPlayVideo(String url) async {
    setState(() {
      _isDownloading = true;
    });

    final uri =
        Uri.parse('http://192.168.140.135:8000/download-and-analyze-youtube/');
    final headers = {'Content-Type': 'application/x-www-form-urlencoded'};
    final body = {'youtube_url': url};

    try {
      // POST request to the FastAPI server
      var response = await http.post(uri, headers: headers, body: body);

      if (response.statusCode == 200) {
        final documentDirectory = await getApplicationDocumentsDirectory();
        File videoFile = File('${documentDirectory.path}/downloaded_video.mp4');
        videoFile.writeAsBytesSync(response.bodyBytes);

        _controller = VideoPlayerController.file(videoFile)
          ..initialize().then((_) {
            setState(() {});
            _controller?.play();
          });
      } else {
        // Handle the case when the server responds with an error
        throw Exception(
            "Failed to download video: HTTP status ${response.statusCode}");
      }
    } catch (e) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text("Error"),
          content: Text("Failed to download video: $e"),
          actions: <Widget>[
            TextButton(
              child: const Text("Ok"),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        ),
      );
    } finally {
      setState(() {
        _isDownloading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Stream Example'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: TextField(
                controller: _urlController,
                decoration: const InputDecoration(
                  labelText: 'Enter YouTube Video URL',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            ElevatedButton(
              onPressed: _isDownloading
                  ? null
                  : () => downloadAndPlayVideo(_urlController.text),
              child: const Text('Download and Analyze Video'),
            ),
            _isDownloading
                ? const CircularProgressIndicator()
                : _controller != null && _controller!.value.isInitialized
                    ? AspectRatio(
                        aspectRatio: _controller!.value.aspectRatio,
                        child: VideoPlayer(_controller!),
                      )
                    : Container(),
          ],
        ),
      ),
    );
  }
}
