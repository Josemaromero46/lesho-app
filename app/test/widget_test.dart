import 'package:flutter_test/flutter_test.dart';

import 'package:lesho_app/main.dart';

void main() {
  testWidgets('La app arranca y muestra la pantalla de inicio',
      (WidgetTester tester) async {
    await tester.pumpWidget(const AppLesho());
    expect(find.text('LESHO'), findsOneWidget);
  });
}
